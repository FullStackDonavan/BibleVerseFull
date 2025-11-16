from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import torch
import os
import uuid
from fastapi.responses import FileResponse

app = FastAPI()

# Allow CORS for any origin (adjust for production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

print("Loading base model and LoRA...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the base anime model
pipe = StableDiffusionPipeline.from_pretrained(
    "hakurei/waifu-diffusion",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)
pipe.to(device)

# Load LoRA from env or default path
lora_model_path = os.getenv("LORA_MODEL_PATH", "./anime_lora.safetensors")
if not os.path.exists(lora_model_path):
    raise RuntimeError(f"LoRA model not found at {lora_model_path}")

lora = PeftModel.from_pretrained(pipe.unet, lora_model_path)
pipe.unet = lora.merge_and_unload()

# Disable NSFW filter
pipe.safety_checker = lambda images, **kwargs: (images, False)

print("Model and LoRA loaded successfully.")

# Output folder
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "generated")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/generate")
async def generate_image(req: GenerateRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    try:
        image = pipe(req.prompt, width=req.width, height=req.height).images[0]
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)

        host_url = os.getenv("HOST_URL", "http://127.0.0.1:5002")
        return {"imageUrl": f"{host_url}/generated/{filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generated/{filename}")
async def serve_image(filename: str):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=True)
