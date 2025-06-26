from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import os
import shutil
import uuid
from real_esrgan_utils import upscale_image

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://image-downloader-oklhz85p7-ardyians-projects.vercel.app/"], 
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Image Upscale API is running!"}


@app.post("/upscale")
async def upscale(image: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    input_path = f"temp/{uuid.uuid4().hex}_{image.filename}"
    output_path = f"temp/upscaled_{uuid.uuid4().hex}_{image.filename}"

    with open(input_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    result = upscale_image(input_path, output_path)
    return FileResponse(result, media_type="image/jpeg")
