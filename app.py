from fastapi import FastAPI
import requests
import uuid
import os

app = FastAPI()

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Hugging Face Inference API
API_URL = "https://api-inference.huggingface.co/models/damo-vilab/text-to-video-ms-1.7b"

HEADERS = {
    "Authorization": f"Bearer {os.getenv('HF_TOKEN')}"
}


@app.post("/generate")
def generate_video(prompt: str):
    response = requests.post(
        API_URL,
        headers=HEADERS,
        json={"inputs": prompt},
        timeout=300
    )

    if response.status_code != 200:
        return {"error": "Video generation failed"}

    filename = f"outputs/{uuid.uuid4()}.mp4"

    with open(filename, "wb") as f:
        f.write(response.content)

    return {
        "message": "Video generated successfully",
        "video_path": filename
    }
