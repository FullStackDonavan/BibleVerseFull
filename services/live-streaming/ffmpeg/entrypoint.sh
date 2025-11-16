#!/bin/sh
set -e

# Example usage:
# ./entrypoint.sh rtmp://nginx-rtmp/live/stream rtmp://upstream-server/app/stream

INPUT_URL=${1:-rtmp://nginx-rtmp/live/stream}
OUTPUT_URL=${2:-rtmp://example.com/live/stream}

echo "Starting FFmpeg relay..."
echo "Input:  $INPUT_URL"
echo "Output: $OUTPUT_URL"

# Relay RTMP stream with minimal CPU usage (copy video/audio streams)
exec ffmpeg -i "$INPUT_URL" \
  -c:v copy -c:a copy \
  -f flv "$OUTPUT_URL"
