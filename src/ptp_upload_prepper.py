#!/usr/bin/env python3
import os
import sys
import shlex
import subprocess
import requests
import time
from pathlib import Path
import mimetypes
import argparse
import logging
import math
import re
import shutil
import csv
import string
from typing import Tuple, List
from unidecode import unidecode
from dotenv import load_dotenv

# Set up logging
logger = logging.getLogger(__name__)
# Env configuration
env_path = Path('.env')
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV = int(os.getenv('ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV', '3'))
CSV_PATH = os.getenv('CSV_PATH', 'films.csv')
DOCKER_CONTAINERS = 'mediainfo,mediainfo-dvd,ffmpeg,pngquant,transmission'.split(',')
GID = os.getenv('GID', '1000')
MEDIA_PATH = os.getenv('MEDIA_PATH')
OUTPUT_PATH = os.getenv('OUTPUT_PATH', 'processed')
PTPIMG_API_KEY = os.getenv('PTPIMG_API_KEY')
TORRENTS_PATH = os.getenv('TORRENTS_PATH', 'torrents')
TRACKER_ANNOUNCE_URL = os.getenv('TRACKER_ANNOUNCE_URL')
TRACKER_SOURCE = os.getenv('TRACKER_SOURCE')
UID = os.getenv('UID', '1000')

# Args
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
"""Main function to handle command-line arguments and process files."""
parser.add_argument("--generate-screens", action="store_true", help="Run only the thumbnail generation functionality")
parser.add_argument("--generate-csv", action="store_true",
                    help="Generate a CSV file list from a directory of torrent files")
parser.add_argument("--torrents-path", default=TORRENTS_PATH, help="Path to the directory containing torrent files")
parser.add_argument("--confirm-hardlinks", action="store_true",
                    help="Prompt for confirmation before creating each hardlink")
parser.add_argument("--csv-path", default=CSV_PATH, help="CSV file containing paths to process")
parser.add_argument("--media-path", default=MEDIA_PATH, required=False, help="Directory where input files are located")
parser.add_argument("--output-path", default=OUTPUT_PATH,
                    help="Directory where output files will be stored (thumbnails, info files, torrents, etc.)")
parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
parser.add_argument("files", nargs="*", help="Files to process")
args = parser.parse_args()


class UploadError(Exception):
    """Custom exception for upload errors."""
    pass


class PtpimgUploader:
    """Handle image uploads to ptpimg.me."""

    def __init__(self, api_key, timeout=None):
        self.api_key = api_key
        self.timeout = timeout
        self.upload_url = "https://ptpimg.me/upload.php"

    def upload_file(self, filename):
        """Upload a file to ptpimg.me and return the image URL(s)."""
        with open(filename, 'rb') as file:
            mime_type, _ = mimetypes.guess_type(filename)
            if not mime_type or not mime_type.startswith('image/'):
                raise ValueError(f'Unsupported file type: {mime_type}')

            files = {'file-upload[]': (Path(filename).name, file, mime_type)}
            data = {'api_key': self.api_key}
            headers = {'referer': 'https://ptpimg.me/index.php'}

            try:
                response = requests.post(self.upload_url, headers=headers, data=data, files=files, timeout=self.timeout)
                response.raise_for_status()
                return [f"https://ptpimg.me/{res['code']}.{res['ext']}" for res in response.json()]
            except requests.RequestException as e:
                raise UploadError(f"Failed to upload {filename}: {str(e)}") from e


def run_command(cmd, docker_image="", timeout=300, check=True, shell=False, **kwargs):
    """Run a shell command with a timeout and handle exceptions."""
    if docker_image:
        # For Docker commands, we always use shell=True to handle complex commands
        shell = True
        # Construct the Docker command, properly escaping the inner command
        inner_cmd = ' '.join(shlex.quote(str(arg)) for arg in cmd)
        cmd = f"docker exec -u {UID}:{GID} {docker_image} sh -c {shlex.quote(inner_cmd)}"
    elif not shell:
        # If not using Docker and shell is False, ensure cmd is a list
        cmd = [str(arg) for arg in cmd] if isinstance(cmd, (list, tuple)) else shlex.split(cmd)

    try:
        result = subprocess.run(cmd, check=check, shell=shell, text=True, capture_output=True, timeout=timeout,
                                **kwargs)
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        handle_logging('error', f"Command timed out after {timeout} seconds: {cmd}")
        raise
    except subprocess.CalledProcessError as e:
        handle_logging('error', f"Command failed: {cmd}")
        handle_logging('error', f"Error output: {e.stderr}")
        raise
    except Exception as e:
        handle_logging('error', f"Unexpected error running command: {cmd}")
        handle_logging('error', f"Error: {str(e)}")
        raise


def check_docker_container_availability():
    """Check if the specified Docker container is available and running."""
    docker_containers_unavailable = DOCKER_CONTAINERS.copy()
    handle_logging('info', f"Checking for Docker containers: {', '.join(docker_containers_unavailable)}")
    try:
        result = subprocess.run(['docker', 'ps', '--format', '{{.Names}}'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            containers = result.stdout.strip().split('\n')
            handle_logging('info', f"Found running Docker containers: {', '.join(containers)}")
            for container in containers:
                if container in docker_containers_unavailable:
                    # log info about the container if verbose flagged
                    handle_logging('info', f"Found running Docker container: {container}")
                    docker_containers_unavailable.remove(container)
            return docker_containers_unavailable
        else:
            handle_logging('warning', "Failed to get list of running Docker containers.")
            return docker_containers_unavailable
    except subprocess.CalledProcessError:
        handle_logging('warning', "Failed to execute Docker command.")
        return docker_containers_unavailable
    except subprocess.TimeoutExpired:
        handle_logging('warning', "Docker check timed out. It might be unresponsive.")
        return docker_containers_unavailable


def check_dependencies():
    """Check if all required dependencies are installed."""
    docker_containers_unavailable = check_docker_container_availability()
    if not docker_containers_unavailable:
        handle_logging('info', "All required Docker containers are available.")
    else:
        handle_logging('error', f"Missing Docker containers: {', '.join(docker_containers_unavailable)}")
        handle_logging('error', "Please start all required Docker containers before running this script.")
        sys.exit(1)


def get_color_matrix(file_path):
    """Determine the color matrix of the video."""
    try:
        output = run_command(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=color_space", "-of",
             "default=noprint_wrappers=1:nokey=1", str(file_path)], docker_image="ffmpeg", timeout=60)
        color_space = output.strip()
        if color_space == "bt709":
            return "bt709"
        elif color_space == "bt601":
            return "bt601"
        elif color_space == "bt2020nc" or color_space == "bt2020c":
            return "bt2020"
        else:
            handle_logging('warning', f"Unrecognized color space: {color_space}. Defaulting to bt709.")
            return "bt709"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        handle_logging('error', f"Failed to determine color matrix for {file_path}: {e}")
        handle_logging('warning', "Defaulting to bt709 color matrix.")
        return "bt709"


def generate_thumbnails(file_path, working_dir, total_runs=6, start_buffer=0.05, end_buffer=0.05, filename_prefix=""):
    """Generate thumbnails for the given video file, handling different color matrices and anamorphic videos."""
    file_path = Path(file_path)
    working_dir = Path(working_dir)
    handle_logging('info', f"Generating thumbnails for {file_path}")

    # Get video duration
    try:
        duration = float(run_command(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             str(file_path)], docker_image="ffmpeg", timeout=60))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        handle_logging('error', f"Failed to get video duration for {file_path}: {e}")
        return []

    # Determine color matrix
    color_matrix = get_color_matrix(file_path)
    handle_logging('info', f"Detected color matrix: {color_matrix}")

    vf = (f"scale=in_h_chr_pos=0:in_v_chr_pos=0:in_color_matrix={color_matrix}:flags=full_chroma_int+full_chroma_inp"
          f"+accurate_rnd+spline") if color_matrix == "bt2020" else (f"scale='max(sar,1)*iw':'max(1/sar,"
                                                                     f"1)*ih':in_h_chr_pos=0:in_v_chr_pos=128"
                                                                     f":in_color_matrix="
                                                                     f"{color_matrix}:flags=full_chroma_int"
                                                                     f"+full_chroma_inp+accurate_rnd+spline")

    # Calculate intervals
    usable_duration = duration * (1 - start_buffer - end_buffer)
    interval = usable_duration / (total_runs - 1)
    start_time = duration * start_buffer

    generated_files = []

    for i in range(total_runs):
        current_time = start_time + (i * interval)
        output_file = working_dir / f"{filename_prefix}{file_path.stem}_thumb_{i + 1}.png"

        try:
            run_command(["ffmpeg", "-y", "-ss", str(current_time), "-i", str(file_path), "-vf", vf, "-pix_fmt", "rgb24",
                         "-vframes", "1", "-q:v", "2", str(output_file)], docker_image="ffmpeg", timeout=120)

            generated_files.append(output_file)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            handle_logging('error', f"Failed to generate thumbnail {i + 1} for {file_path}: {e}")

    # Optimize generated images
    for file in generated_files:
        try:
            run_command(["pngquant", "--force", "--output", str(file), "--", str(file)], docker_image="pngquant",
                        timeout=60)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            handle_logging('error', f"Failed to optimize thumbnail {file}: {e}")
            handle_logging('info', f"Generated {len(generated_files)} thumbnails for {file_path}")
    return generated_files


def get_dvd_mediainfo(file_path):
    """Get mediainfo output for DVD."""
    file_path = Path(file_path)

    try:
        info_output = run_command(['mediainfo', str(file_path)], docker_image="mediainfo-dvd", timeout=60)
        return info_output.replace(str(file_path), file_path.name)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        handle_logging('error', f"Failed to run mediainfo in Docker container: {e}")
        return None


def create_dvd_info_file(vob_file: Path, file_object) -> Tuple[bool, str]:
    """Create an info file with media information for DVD structure."""
    handle_logging('info', f"Creating media info for DVD: {vob_file}")

    try:
        # Get mediainfo for VOB file
        vob_info = get_dvd_mediainfo(vob_file)
        if vob_info is None:
            return True, "Failed to get mediainfo for VOB file"

        # Get corresponding IFO file
        ifo_file = vob_file.parent / f"{vob_file.stem[:-1]}0.IFO"
        if not ifo_file.exists():
            return True, f"Corresponding IFO file not found: {ifo_file}"

        # Get mediainfo for IFO file
        ifo_info = get_dvd_mediainfo(ifo_file)
        if ifo_info is None:
            return True, "Failed to get mediainfo for IFO file"

        # Write mediainfo to file
        file_object.write(f"Media Info for VOB file {vob_file.name}:\n")
        file_object.write("=" * 50 + "\n")
        file_object.write(vob_info)
        file_object.write("\n\n")
        file_object.write(f"Media Info for IFO file {ifo_file.name}:\n")
        file_object.write("=" * 50 + "\n")
        file_object.write(ifo_info)

        handle_logging('info', f"Added media info for DVD: {vob_file}")
        return False, ""

    except Exception as e:
        handle_logging('error', f"Failed to create media info for DVD {vob_file}: {e}")
        return True, f"Failed to create media info: {str(e)}"


def create_info_file(file_path: Path, file_object) -> Tuple[bool, str]:
    """
    Create an info file with media information for the given file.

    Returns:
    - Tuple[bool, str]: (is_skipped, skip_reason)
    """
    handle_logging('info', f"Creating media info for {file_path}")

    try:
        info_output = get_dvd_mediainfo(file_path)
        if info_output is None:
            return True, f"Failed to get mediainfo for {file_path}"

        # Check for skip criteria
        if "IsTruncated : Yes" in info_output:
            return True, "File is truncated"

        if "Format                                   : MPEG-4" in info_output:
            return True, "File is in MP4 format"

        # Check for HEVC in non-HDR content
        is_hevc = "Format                                   : HEVC" in info_output
        is_uhd = "Width                                    : 3840" in info_output or ("Height                          "
                                                                                      "         : 2160") in info_output
        is_hdr = "HDR format                               : " in info_output

        if is_hevc and not (is_uhd or is_hdr):
            return True, "HEVC used for non-HDR, non-UHD content"

        file_object.write(info_output)
        handle_logging('info', f"Added media info for: {file_path}")
        return False, ""

    except Exception as e:
        handle_logging('error', f"Failed to create media info for {file_path}: {e}")
        return True, f"Failed to create media info: {str(e)}"


def upload_images(working_dir, output_file):
    working_dir = Path(working_dir)
    output_file = Path(output_file)
    handle_logging('info', f"Uploading images from {working_dir}")

    uploader = PtpimgUploader(PTPIMG_API_KEY)
    with output_file.open('a') as f:
        f.write("\n\n")

    for img_path in working_dir.glob("*.[jp][pn]g"):
        for attempt in range(3):  # 3 retries
            try:
                urls = uploader.upload_file(str(img_path))
                with output_file.open('a') as f:
                    for url in urls:
                        f.write(f"[img]{url}[/img]\n")
                handle_logging('info', f"Uploaded {img_path} to ptpimg.me")
                break
            except UploadError as e:
                handle_logging('warning', f"Upload attempt {attempt + 1} failed for {img_path}: {e}")
                if attempt == 2:  # Last attempt
                    handle_logging('error', f"Failed to upload {img_path} after 3 attempts")
                else:
                    time.sleep(5)  # Wait before retrying


def get_total_size(path):
    """Calculate the total size of a file or directory."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    elif path.is_dir():
        return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    else:
        raise ValueError(f"Path {path} is neither a file nor a directory")


def calculate_piece_size(total_size):
    """Calculate the appropriate piece size based on the total size."""
    piece_sizes = [(16, 16 * 1024),  # 16 KB
                   (32, 32 * 1024),  # 32 KB
                   (64, 64 * 1024),  # 64 KB
                   (128, 128 * 1024),  # 128 KB
                   (256, 256 * 1024),  # 256 KB
                   (512, 512 * 1024),  # 512 KB
                   (1024, 1 * 1024 * 1024),  # 1 MB
                   (2048, 2 * 1024 * 1024),  # 2 MB
                   (4096, 4 * 1024 * 1024),  # 4 MB
                   (8192, 8 * 1024 * 1024),  # 8 MB
                   (16384, 16 * 1024 * 1024)  # 16 MB
                   ]

    # For very small files, use the smallest piece size
    if total_size < piece_sizes[0][1]:
        return piece_sizes[0][1]

    for piece_count, piece_size in piece_sizes:
        pieces = total_size / piece_size
        if pieces <= 2000:
            # If we're already under 1000 pieces, and we can go smaller, do so
            if pieces < 1000 and piece_size > piece_sizes[0][1]:
                return piece_sizes[piece_sizes.index((piece_count, piece_size)) - 1][1]
            return piece_size

    # If we're here, the file is very large. Use the maximum piece size.
    return piece_sizes[-1][1]


def create_torrent(file_path, torrent_file):
    """Create a torrent file for the given file or directory."""
    file_path = Path(file_path)
    handle_logging('info', f"Creating torrent file for {file_path}")

    try:
        total_size = get_total_size(file_path)
        piece_size = calculate_piece_size(total_size)
        piece_count = math.ceil(total_size / piece_size)

        handle_logging('info', f"Total size: {total_size} bytes")
        handle_logging('info', f"Selected piece size: {piece_size} bytes")
        handle_logging('info', f"Estimated piece count: {piece_count}")

        # Use the sanitized file_path
        sanitized_path, _ = process_path(file_path)

        run_command(['transmission-create', '--private', '--tracker', TRACKER_ANNOUNCE_URL, '--source', TRACKER_SOURCE,
                     '--piece-size', str(piece_size), '--outfile', str(torrent_file), str(file_path)],
                    docker_image="transmission", timeout=300)  # 5 minute timeout for torrent creation
        handle_logging('info', f"Created torrent file: {torrent_file}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        handle_logging('error', f"Failed to create torrent file for {file_path}: {e}")
    except ValueError as e:
        handle_logging('error', f"Error calculating torrent properties: {e}")


def is_bluray_structure(directory):
    """Check if the directory has a Blu-ray structure."""
    directory = Path(directory)
    return (directory / "BDMV").is_dir()


def is_dvd_structure(directory):
    """Check if the directory has a DVD structure."""
    directory = Path(directory)
    return (directory / "VIDEO_TS").is_dir()


def find_main_movie_file(directory):
    """Find the main movie file in a Blu-ray or DVD structure."""
    directory = Path(directory)
    if is_bluray_structure(directory):
        stream_dir = directory / "BDMV" / "STREAM"
        if stream_dir.is_dir():
            m2ts_files = list(stream_dir.glob("*.m2ts"))
            if m2ts_files:
                return max(m2ts_files, key=lambda f: f.stat().st_size)
    elif is_dvd_structure(directory):
        vob_files = list(directory.glob("VIDEO_TS/*.VOB"))
        if vob_files:
            return max(vob_files, key=lambda f: f.stat().st_size)
    return None


def process_file(file_path, media_path, generate_screens_only=False, confirm_hardlinks=False, output_path=None) -> \
Tuple[bool, str]:
    file_path = Path(media_path) / Path(file_path)
    handle_logging('info', f"Processing: {file_path}")
    output_path = output_path or Path.cwd()

    if file_path.is_dir():
        if is_bluray_structure(file_path):
            handle_logging('info', f"Detected Blu-ray structure: {file_path}")
            return process_disc_structure(file_path, media_path, generate_screens_only, confirm_hardlinks, output_path)
        elif is_dvd_structure(file_path):
            handle_logging('info', f"Detected DVD structure: {file_path}")
            return process_disc_structure(file_path, media_path, generate_screens_only, confirm_hardlinks, output_path)
        elif is_authorized_multifile_structure(file_path):
            handle_logging('info', f"Detected authorized multi-file structure: {file_path}")
            return process_authorized_multifile(file_path, media_path, generate_screens_only, confirm_hardlinks,
                                                output_path)
        else:
            handle_logging('warning', f"Unrecognized directory structure: {file_path}")
            return False, f"Directory is not a recognized structure: {file_path}"
    elif file_path.is_file():
        if file_path.suffix.lower() == '.iso':
            handle_logging('warning', f"ISO file detected: {file_path}. Skipping processing.")
            return False, "ISO files are not supported for processing"
        handle_logging('info', f"Processing single file: {file_path}")
        return process_single_file(file_path, media_path, generate_screens_only, confirm_hardlinks, output_path)
    else:
        handle_logging('error', f"File not found: {file_path}")
        return False, f"File not found: {file_path}"


def process_disc_structure(file_path, media_path, generate_screens_only, confirm_hardlinks, output_path):
    is_bluray = is_bluray_structure(file_path)
    handle_logging('info', f"Detected {'Blu-ray' if is_bluray else 'DVD'} structure: {file_path}")

    main_movie_file = find_main_movie_file(file_path)
    if not main_movie_file:
        return False, f"Could not find main movie file in {file_path}"

    safe_path, needs_sanitization = process_path(file_path.relative_to(media_path))
    safe_path = media_path / safe_path

    if needs_sanitization:
        handle_logging('warning', f"Unsafe directory name detected: {file_path}")
        safe_path = create_safe_directory(file_path, media_path, confirm_hardlinks)
        if safe_path is None:
            return False, f"Skipping processing of {file_path} due to unsafe directory name"
        file_path = safe_path

    working_dir = output_path / safe_path.name
    working_dir.mkdir(parents=True, exist_ok=True)

    info_file = working_dir / "info.txt"

    if not generate_screens_only:
        with open(info_file, 'w') as f:
            if is_bluray:
                is_skipped, skip_reason = create_info_file(main_movie_file, f)
            else:  # DVD
                is_skipped, skip_reason = create_dvd_info_file(main_movie_file, f)
            if is_skipped:
                return False, skip_reason

    generate_thumbnails(main_movie_file, working_dir)

    if not generate_screens_only:
        upload_images(working_dir, info_file)
        create_torrent(file_path, working_dir / f"{file_path.name}.torrent")

    handle_logging('info', f"Completed processing of {file_path}")
    return True, ""


def process_authorized_multifile(file_path, media_path, generate_screens_only, confirm_hardlinks, output_path):
    handle_logging('info', f"Detected authorized multi-file structure: {file_path}")

    mkv_files = list(Path(file_path).glob('*.mkv'))
    if not mkv_files:
        return False, f"Could not find any MKV files in {file_path}"

    safe_path, dir_needs_sanitization = process_path(Path(file_path).relative_to(media_path))

    # Check if any files need sanitization
    files_need_sanitization = any(process_path(f.name)[1] for f in mkv_files)

    if dir_needs_sanitization or files_need_sanitization:
        handle_logging('warning', f"Unsafe directory name or filenames detected: {file_path}")
        safe_path = create_safe_directory_with_files(file_path, media_path, confirm_hardlinks)
        if safe_path is None:
            return False, f"Skipping processing of {file_path} due to unsafe directory name or filenames"

        # Re-find the MKV files in the new safe directory
        mkv_files = list(safe_path.glob('*.mkv'))
        if not mkv_files:
            return False, f"Could not find any MKV files in {safe_path} after sanitization"
    else:
        safe_path = Path(file_path)

    working_dir = output_path / safe_path.stem
    working_dir.mkdir(parents=True, exist_ok=True)

    info_file = working_dir / "info.txt"

    if not generate_screens_only:
        with open(info_file, 'w') as f:
            for mkv_file in mkv_files:
                f.write(f"Media Info for {mkv_file.name}:\n")
                f.write("=" * 50 + "\n")
                is_skipped, skip_reason = create_info_file(mkv_file, f)
                if is_skipped:
                    return False, skip_reason
                f.write("\n\n")

    for mkv_file in mkv_files:
        filename_prefix = f"{mkv_file.stem}_"
        generate_thumbnails(mkv_file, working_dir, filename_prefix=filename_prefix)

    if not generate_screens_only:
        upload_images(working_dir, info_file)
        create_torrent(safe_path, working_dir / f"{safe_path.stem}.torrent")

    handle_logging('info', f"Completed processing of {file_path}")
    return True, ""


def process_single_file(file_path, media_path, generate_screens_only, confirm_hardlinks, output_path):
    safe_path, needs_sanitization = process_path(file_path.relative_to(media_path))
    safe_path = media_path / safe_path

    if needs_sanitization:
        handle_logging('warning', f"Unsafe filename detected: {file_path}")
        safe_path = create_safe_hardlink(file_path, media_path, confirm_hardlinks)
        if safe_path is None:
            return False, f"Skipping processing of {file_path} due to unsafe filename"
        file_path = safe_path

    working_dir = output_path / safe_path.stem
    working_dir.mkdir(parents=True, exist_ok=True)

    info_file = working_dir / "info.txt"

    if not generate_screens_only:
        with open(info_file, 'w') as f:
            is_skipped, skip_reason = create_info_file(file_path, f)
        if is_skipped:
            return False, skip_reason

    generate_thumbnails(file_path, working_dir)

    if not generate_screens_only:
        upload_images(working_dir, info_file)
        create_torrent(file_path, working_dir / f"{safe_path.stem}.torrent")

    handle_logging('info', f"Completed processing of {file_path}")
    return True, ""


def create_safe_hardlink(original_file, target_dir, confirm=False):
    """Create a hardlink with a safe filename."""
    original_file = Path(original_file)
    target_dir = Path(target_dir)

    safe_relative_path, _ = process_path(original_file.name)
    safe_path = target_dir / safe_relative_path

    if confirm:
        user_input = input(f"Create hardlink: {original_file} -> {safe_path}? (y/n): ")
        if user_input.lower() != 'y':
            handle_logging('info', "Hardlink creation skipped by user.")
            return None

    try:
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.unlink(missing_ok=True)
        os.link(original_file, safe_path)
        handle_logging('info', f"Created safe hardlink: {safe_path}")
        return safe_path
    except OSError as e:
        handle_logging('error', f"Failed to create hardlink for {original_file}: {e}")
        return None


def create_safe_directory(original_dir, media_path, confirm=False):
    """Create a new directory with a safe name and copy disc structure."""
    original_dir = Path(original_dir)
    media_path = Path(media_path)
    relative_path = original_dir.relative_to(media_path)
    safe_relative_path, _ = process_path(relative_path)
    safe_dir = media_path / safe_relative_path

    if confirm:
        user_input = input(f"Create safe directory: {original_dir} -> {safe_dir}? (y/n): ")
        if user_input.lower() != 'y':
            handle_logging('info', "Safe directory creation skipped by user.")
            return None

    try:
        # Overwrite the directory if it already exists
        shutil.rmtree(safe_dir, ignore_errors=True)
        safe_dir.mkdir(parents=True, exist_ok=True)
        handle_logging('info', f"Created safe directory: {safe_dir}")

        # Copy the entire directory structure
        for src_dir, dirs, files in os.walk(original_dir):
            dst_dir = safe_dir / Path(src_dir).relative_to(original_dir)
            try:
                dst_dir.mkdir(exist_ok=True)
            except OSError as e:
                handle_logging('error', f"Failed to create directory {dst_dir}: {e}")
                return None

            for file_ in files:
                src_file = Path(src_dir) / file_
                dst_file = dst_dir / file_
                try:
                    shutil.copy2(src_file, dst_file, follow_symlinks=False)
                except IOError as e:
                    handle_logging('error', f"Failed to copy file {src_file} to {dst_file}: {e}")
                    return None

        handle_logging('info', f"Copied disc structure from {original_dir} to {safe_dir}")
        return safe_dir
    except OSError as e:
        handle_logging('error', f"Failed to create safe directory for {original_dir}: {e}")
        return None


def create_safe_directory_with_files(original_dir, media_path, confirm=False):
    """Create a new directory with a safe name and hardlink files with safe names."""
    original_dir = Path(original_dir)
    media_path = Path(media_path)
    relative_path = original_dir.relative_to(media_path)
    safe_relative_path, _ = process_path(relative_path)
    safe_dir = media_path / safe_relative_path

    if confirm:
        user_input = input(f"Create safe directory: {original_dir} -> {safe_dir}? (y/n): ")
        if user_input.lower() != 'y':
            handle_logging('info', "Safe directory creation skipped by user.")
            return None

    try:
        safe_dir.mkdir(parents=True, exist_ok=True)
        handle_logging('info', f"Created safe directory: {safe_dir}")

        for file in original_dir.iterdir():
            if file.is_file():
                safe_file = create_safe_hardlink(file, safe_dir, confirm)
                if safe_file is None:
                    handle_logging('warning', f"Failed to create safe hardlink for {file}")

        return safe_dir
    except OSError as e:
        handle_logging('error', f"Failed to create safe directory for {original_dir}: {e}")
        return None


def is_authorized_multifile_structure(directory):
    """Check if the directory contains ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV or fewer MKV files and no other file
    types."""
    all_files = list(directory.iterdir())
    mkv_files = [f for f in all_files if f.is_file() and f.suffix.lower() == '.mkv']

    return (0 < len(mkv_files) <= ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV and len(all_files) == len(mkv_files))


def is_safe_filename(filename):
    """Check if a filename is safe for cross-platform use."""
    safe_chars = set(string.ascii_letters + string.digits + ".,+-_'()[]{} ")
    return all(char in safe_chars for char in filename)


def sanitize_filename(filename):
    """Sanitize a filename to make it safe for cross-platform use."""
    filename = unidecode(filename)
    safe_chars = set(string.ascii_letters + string.digits + ".,+-_'()[]{} ")
    sanitized = ''.join(char if char in safe_chars else '.' for char in filename)
    sanitized = sanitized.strip('.')
    sanitized = re.sub(r'\.{2,}', '.', sanitized)
    return sanitized


def process_path(path):
    """Process a path, sanitizing each component if necessary."""
    path_parts = Path(path).parts
    sanitized_parts = []
    needs_sanitization = False

    for part in path_parts:
        if is_safe_filename(part):
            sanitized_parts.append(part)
        else:
            sanitized_parts.append(sanitize_filename(part))
            needs_sanitization = True

    return Path(*sanitized_parts), needs_sanitization


def parse_torrent_file(torrent_path):
    """Parse a torrent file and extract relevant information."""
    torrent_path = Path(torrent_path)
    handle_logging('info', f"Parsing torrent file: {torrent_path}")

    try:
        output = run_command(['transmission-show', str(torrent_path)], docker_image="transmission")
    except subprocess.CalledProcessError as e:
        handle_logging('error', f"Failed to parse torrent file {torrent_path}: {e}")
        return None, None, None, 0, False, False

    name_match = re.search(r'Name: (.*)', output)
    name = name_match.group(1) if name_match else ""

    comment_match = re.search(r'Comment: (.*)', output)
    comment = comment_match.group(1) if comment_match else ""

    file_section = re.search(r'FILES\n(.*)', output, re.DOTALL)
    if file_section:
        file_lines = file_section.group(1).strip().split('\n')
        files = []
        for line in file_lines:
            # Use a non-greedy match to find the last occurrence of the file size pattern
            match = re.match(r'(.+?) \([\d.]+ [KMGT]?B\)$', line)
            if match:
                files.append(match.group(1).strip())

        if files:
            if len(files) == 1:
                filepath = files[0]
                _, needs_sanitization = process_path(filepath)
                return name, comment, filepath, 1, True, not needs_sanitization
            else:
                common_path = os.path.commonpath(files)
                is_bluray = any('BDMV' in f for f in files)
                is_dvd = any('VIDEO_TS' in f for f in files)

                if is_bluray or is_dvd:
                    filepath = os.path.dirname(common_path)  # Use parent directory for disc structures
                else:
                    filepath = common_path

                needs_sanitization = any(process_path(f)[1] for f in files)
                return name, comment, filepath.rstrip('/'), len(files), False, not needs_sanitization

    handle_logging('warning', f"Could not parse file information from torrent: {torrent_path}")
    return name, comment, None, 0, False, False


def generate_csv(torrent_directory, output_file):
    """Generate a CSV file list from a directory of torrent files."""
    torrent_directory = Path(torrent_directory)
    output_file = Path(output_file)
    handle_logging('info', f"Generating file list from torrent directory: {torrent_directory}")

    try:
        # Collect all entries
        entries = []
        for torrent_file in torrent_directory.glob('*.torrent'):
            name, comment, filepath, num_files, is_single_file, is_safe = parse_torrent_file(torrent_file)
            if filepath:
                entries.append([name, comment, filepath, num_files, str(is_single_file).lower(), str(is_safe).lower()])
                if not is_single_file and not is_safe:
                    handle_logging('warning', (
                        f"Multi-file torrent with unsafe filenames detected: {filepath}. Manual processing required."))
                elif not is_safe:
                    handle_logging('info',
                                   f"Single-file torrent with unsafe filename detected: {filepath}. Will attempt to "
                                   f"create safe hardlink during processing.")
        # Sort entries: single-file torrents first, then multi-file torrents
        entries.sort(key=lambda x: x[4] == 'false')

        # Write sorted entries to CSV
        with output_file.open('w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            csvwriter.writerow(['name', 'comment', 'filepath', 'number_of_files', 'single_file', 'safe_filename'])
            csvwriter.writerows(entries)

        handle_logging('info', f"Generated CSV file list: {output_file}")
        handle_logging('info', f"Total entries: {len(entries)}")
        handle_logging('info', f"Single-file entries: {sum(1 for entry in entries if entry[4] == 'true')}")
        handle_logging('info', f"Multi-file entries: {sum(1 for entry in entries if entry[4] == 'false')}")
    except IOError as e:
        handle_logging('error', f"Failed to write CSV file {output_file}: {e}")
        sys.exit(1)


def handle_logging(level, message):
    """Handle logging based on the given level and message."""
    if level == "info":
        if args.verbose:
            logger.info(message)
    elif level == "warning":
        if args.verbose:
            logger.warning(message)
    elif level == "error":
        logger.error(message)
    elif level == "debug":
        if args.verbose:
            logger.debug(message)
    else:
        if args.verbose:
            logger.info(message)


def main():
    # update the global logger
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Load environment variables
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        handle_logging('info', f"Loaded environment variables from {env_path}")
    else:
        handle_logging('warning', f".env file not found at {env_path}. Using default environment variables.")

    # Validate required environment variables
    # log out the env variables
    if not PTPIMG_API_KEY or not TRACKER_ANNOUNCE_URL:
        handle_logging('error', "PTPIMG_API_KEY and TRACKER_ANNOUNCE_URL must be set in the .env file or environment.")
        sys.exit(1)

    # Log configuration
    handle_logging('info', f"TRACKER_SOURCE: {TRACKER_SOURCE}")
    handle_logging('info', f"ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV: {ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV}")

    if not args.media_path and not args.generate_csv:
        parser.print_help()
        sys.exit(1)

    handle_logging('info', f"Using tracker source: {TRACKER_SOURCE}")
    handle_logging('info', f"Root path: {args.media_path}")

    check_dependencies()

    if args.generate_csv:
        torrent_directory = Path(args.torrents_path)
        csv_path = Path(args.csv_path)
        generate_csv(torrent_directory, csv_path)
        return

    if args.csv_path:
        try:
            with open(args.csv_path, 'r', newline='') as csvfile:
                csvreader = csv.DictReader(csvfile)
                file_paths = [row['filepath'] for row in csvreader]
        except Exception as e:
            handle_logging('error', f"Error reading CSV file {args.csv_path}: {e}")
            sys.exit(1)
    else:
        file_paths = args.files

    if not file_paths:
        parser.print_help()
        sys.exit(1)

    successful_count = 0
    error_count = 0
    skipped_files: List[Tuple[str, str]] = []
    output_path = Path(args.output_path) if args.output_path else Path.cwd()
    output_path.mkdir(parents=True, exist_ok=True)
    handle_logging('info', f"Output directory: {output_path}")

    for file_path in file_paths:
        try:
            success, message = process_file(file_path, args.media_path, args.generate_screens, args.confirm_hardlinks,
                                            output_path)
            if success:
                successful_count += 1
            else:
                error_count += 1
                skipped_files.append((file_path, message))
        except Exception as e:
            handle_logging('error', f"Error processing {file_path}: {e}")
            skipped_files.append((file_path, str(e)))
            error_count += 1

    handle_logging('info', "Script execution completed.")
    handle_logging('info', f"Successfully processed: {successful_count}")
    handle_logging('info', f"Errors/Skipped: {error_count}")

    if skipped_files:
        handle_logging('error', "Skipped files:")
        for file_path, reason in skipped_files:
            handle_logging('error', f"  - {file_path}: {reason}")


if __name__ == "__main__":
    main()
