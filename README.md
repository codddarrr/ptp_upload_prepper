# PTP Upload Prepper

PTP Upload Prepper is a Python script that streamlines video file processing, torrent creation, and image uploading for
PassThePopcorn (PTP) users. It's designed for cross-platform compatibility, including support for network drives.

## Quick Start

1. Clone the repository:
   ```sh
   git clone https://github.com/codddarrr/ptp_upload_prepper.git
   cd ptp_upload_prepper
   ```
2. Set up the environment:
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. Configure settings via `.env` file: (see [Configuration](#configuration) for all options)
   ```sh
   cp .env.dist .env
   # Edit .env with your settings
   ```
4. Make sure that the `CSV_PATH`, `MEDIA_PATH`, `OUTPUT_PATH`, and `TORRENTS_PATH` you configured in step 3 exist, then
   start the Docker containers:
   ```sh
   docker-compose up -d
   ```
5. You're ready to go! Follow these steps to process your files:

   **a.** Download the torrents you want to upload to PTP.  
   **b.** Save the media files to your `MEDIA_PATH` directory and the torrent files to your `TORRENTS_PATH` directory.  
   **c.** Generate a CSV file list, which will be created at `CSV_PATH`:
   ```sh
   python src/ptp_upload_prepper.py --generate-csv # Add --verbose for detailed output
   ```
   **d.** Process the files in the CSV, which will produce output files in the `OUTPUT_PATH` directory:
   ```sh
   python src/ptp_upload_prepper.py # Add --verbose for detailed output
   ```

The script will process the files listed in the CSV, performing the following tasks:

- Generate screenshots (correctly accounting for anamorphic videos, color spaces, etc.)
- Create media info files (including special handling for DVD structures)
- Upload screenshots to PTPImg using your `PTPIMG_API_KEY`
- Create torrent files with tracker information based on `TRACKER_ANNOUNCE_URL` and `TRACKER_SOURCE` (using the correct
  piece size for PTP)

After processing, you'll find a directory for each file in the `OUTPUT_PATH`, containing:

- `images.png`: Uploaded images (already on PTPImg)
- `info.txt`: Media information and BBCode for images
- `torrent_file.torrent`: Torrent file with tracker information

Remember to review each processed file before uploading to PTP. This script aims to significantly reduce the time needed
for upload preparation.

## Table of Contents

- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Docker Containers](#docker-containers)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Key Features

- Automated thumbnail generation for video files
- Comprehensive media info file creation (including special handling for DVD structures)
- Seamless image upload to PTPImg with BBCode generation
- Torrent file creation with customizable tracker information
- CSV generation from existing torrent files
- Cross-platform filename sanitization
- Safe file handling through hardlink creation
- Flexible root path specification for input files

## Prerequisites

- Docker and Docker Compose
- Python 3.6 or higher

## Installation

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment.
3. Install required Python packages: `pip install -r requirements.txt`
4. Configure the settings (see [Configuration](#configuration)).
5. Start the Docker containers: `docker-compose up -d`

## Configuration

Copy `.env.dist` to `.env` and update it with your settings:

```env
ALLOWED_NUMBER_OF_FILES_IN_MULTI_MKV=3
CSV_PATH=./films.csv
GID=$(id -g)
MEDIA_PATH=
OUTPUT_PATH=./processed
PTPIMG_API_KEY=
TORRENTS_PATH=./torrents
TRACKER_ANNOUNCE_URL=
TRACKER_SOURCE=
UID=$(id -u)
```

Ensure you set the correct `MEDIA_PATH`, `PTPIMG_API_KEY`, and `TRACKER_ANNOUNCE_URL`.

## Usage

### Generate CSV from Existing Torrents

```sh
python src/ptp_upload_prepper.py --generate-csv
```

### Process Files

To process files listed in the CSV:

```sh
python src/ptp_upload_prepper.py
```

To process specific files:

```sh
python src/ptp_upload_prepper.py /path/to/file1 /path/to/file2
```

### Generate Screenshots Only

```sh
python src/ptp_upload_prepper.py --generate-screens /path/to/file
```

### Options

- `--generate-csv`: Generate a CSV file list from the torrent files directory
- `--generate-screens`: Run only the thumbnail generation functionality
- `--confirm-hardlinks`: Prompt for confirmation before creating each hardlink
- `--verbose`: Enable detailed logging

## Docker Containers

The script utilizes these Docker containers:

- `ffmpeg`: Video processing and thumbnail generation
- `mediainfo`: General media information extraction
- `mediainfo-dvd`: DVD-specific media information
- `pngquant`: PNG image optimization
- `transmission`: Torrent file creation and parsing

These containers are defined in `docker-compose.yml` and start automatically with `docker-compose up -d`.

## Notes

- Filenames are sanitized for cross-platform compatibility.
- Hardlinks are created in the same root path as input files, preserving the directory structure.
- Output files (thumbnails, info files, etc.) are generated in the `OUTPUT_PATH` directory.
- A summary of successfully processed files and any errors/skipped files is provided at the end of each run.

## Troubleshooting

- Use the `--verbose` flag for detailed logging during troubleshooting.
- Ensure all Docker containers are running: `docker-compose ps`
- Check Docker container logs: `docker-compose logs <container_name>`