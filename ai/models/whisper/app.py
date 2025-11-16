from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import tempfile
import os
import traceback

app = FastAPI()

# Allow CORS for testing and cross-service calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Whisper model...")
device = "cuda" if whisper.torch.cuda.is_available() else "cpu"
model = whisper.load_model("base").to(device)
print("Model loaded.")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp3", ".wav", ".m4a")):
        raise HTTPException(status_code=400, detail="Invalid file type. Must be MP3, WAV, or M4A.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = model.transcribe(tmp_path)
        os.remove(tmp_path)

        return {"transcription": result["text"]}

    except Exception as e:
        error_message = f"Error during transcription: {str(e)}\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_message)
