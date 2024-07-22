# Use Alpine as the base image for a small footprint
FROM alpine:latest

# Install pngquant and tini
RUN apk add --no-cache pngquant tini

# Set tini as the entry point
ENTRYPOINT ["/sbin/tini", "--"]

# Default command to keep the container running
CMD ["tail", "-f", "/dev/null"]