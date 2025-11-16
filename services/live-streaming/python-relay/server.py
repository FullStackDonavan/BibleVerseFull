import asyncio
import json
import subprocess
from aiohttp import web
from aiortc import RTCPeerConnection, MediaStreamTrack
from aiortc.contrib.media import MediaRelay

relay = MediaRelay()
pcs = set()

RTMP_URL = "rtmp://nginx-rtmp/live/stream"

async def offer(request):
    params = await request.json()
    offer_sdp = params["sdp"]
    offer_type = params["type"]

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    async def on_track(track: MediaStreamTrack):
        print(f"Track received: {track.kind}")

        # Pipe the track to ffmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "yuv420p",
            "-s", "640x480",
            "-r", "30",
            "-i", "-",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-f", "flv",
            RTMP_URL
        ]
        process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        
        while True:
            frame = await track.recv()
            process.stdin.write(frame.to_ndarray().tobytes())

    # Set remote description
    await pc.setRemoteDescription(
        aiortc.RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    )

    # Create answer
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.json_response(
        {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
    )

async def on_shutdown(app):
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

app = web.Application()
app.router.add_post("/offer", offer)
app.on_shutdown.append(on_shutdown)

web.run_app(app, port=3001)
