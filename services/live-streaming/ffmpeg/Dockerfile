# Use a lightweight Debian-based image with FFmpeg
FROM jrottenberg/ffmpeg:5.1-ubuntu

# Install any extras if needed
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy your relay scripts (if needed)
# COPY ./relay.py .

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]


# Set default command to keep container alive (or run FFmpeg dynamically)
CMD ["tail", "-f", "/dev/null"]


