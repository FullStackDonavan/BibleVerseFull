from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from diffusers import StableDiffusionXLPipeline  # Note the SDXL-specific pipeline import
import torch
import os
import uuid

app = Flask(__name__)
CORS(app)

# Load the SDXL 1.0 pipeline (huggingface model repo)
# Make sure you have accepted the model license and have access on HF!
model_id = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    revision="fp16",
)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Disable safety checker (for dev only)
pipe.safety_checker = lambda images, **kwargs: (images, False)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data.get("prompt", "")
    width = data.get("width", 1024)   # SDXL default size is 1024x1024 or similar
    height = data.get("height", 1024)

    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        # SDXL uses a two-stage process: prompt and negative prompt. You can customize as needed.
        # For simplicity, no negative prompt here.
        image = pipe(prompt).images[0]

        output_dir = os.path.join(os.path.dirname(__file__), "generated")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)

        return jsonify({
            "imageUrl": f"http://127.0.0.1:5002/generated/{filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generated/<filename>')
def serve_image(filename):
    output_dir = os.path.join(os.path.dirname(__file__), "generated")
    return send_from_directory(output_dir, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
