from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from diffusers import StableVideoDiffusionPipeline
import torch
import os
import uuid
from PIL import Image
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the SVD model once on startup
print("Loading Stable Video Diffusion model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to(device)
print("Model loaded.")

@app.post("/generate-video")
async def generate_video(file: UploadFile = File(...)):
    # Check if file is image
    if not file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
        raise HTTPException(status_code=400, detail="Please upload a valid image file.")

    try:
        # Load the uploaded image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run video generation
        frames = pipe(image, num_frames=14).frames  # List of PIL images

        # Save as MP4
        video_filename = f"{uuid.uuid4().hex}.mp4"
        video_path = os.path.join(OUTPUT_DIR, video_filename)
        frames[0].save(
            video_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // 25,  # ~25 FPS
            loop=0
        )

        return {"videoUrl": f"http://127.0.0.1:5003/generated_videos/{video_filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generated_videos/{filename}")
async def get_video(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(filepath, media_type="video/mp4")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5003, reload=True)
