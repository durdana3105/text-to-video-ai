from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from fastapi import FastAPI
from diffusers import DiffusionPipeline
import torch
import os
import uuid
import imageio

app = FastAPI()
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load pretrained text-to-video model
pipe = DiffusionPipeline.from_pretrained(
    "damo-vilab/text-to-video-ms-1.7b",
    torch_dtype=torch.float16
)

# Move model to GPU
pipe.to("cuda")


@app.post("/generate")
def generate_video(prompt: str):
    # Generate video frames
    result = pipe(prompt, num_frames=16)
    frames = result.frames[0]

    # Save video
    filename = f"outputs/{uuid.uuid4()}.mp4"
    imageio.mimsave(filename, frames, fps=8)

    return {
        "message": "Video generated successfully",
        "video_path": filename
    }
