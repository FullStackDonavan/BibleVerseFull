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

# Allow CORS for any origin (adjust in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body schema
class GenerateRequest(BaseModel):
    prompt: str
    width: int = 512
    height: int = 512

# Load base model and LoRA adapter once on startup
print("Loading models...")
device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = StableDiffusionPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipe.to(device)

lora_model_path = os.getenv("LORA_MODEL_PATH", "./ume_modern_pixelart.safetensors")
lora = PeftModel.from_pretrained(pipe.unet, lora_model_path)
pipe.unet = lora.merge_and_unload()

pipe.safety_checker = lambda images, **kwargs: (images, False)
print("Models loaded.")

# Ensure generated directory exists
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

        return {"imageUrl": f"http://127.0.0.1:5002/generated/{filename}"}

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
