from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import sys
from uuid import uuid4
import uvicorn

from src.pipeline.faceswap_pipeline import initiate_face_swapper
from src.logger import logging
from src.exceptions import CustomException

app = FastAPI()

# === Add CORS middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific domain(s) in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/face-swap")
async def face_swap(
    multi_face_img: UploadFile = File(...),
    single_face_img: UploadFile = File(...)
):
    try:
        # Save uploaded images to disk
        multi_path = os.path.join(UPLOAD_DIR, f"multi_{uuid4().hex}.jpg")
        single_path = os.path.join(UPLOAD_DIR, f"single_{uuid4().hex}.jpg")

        with open(multi_path, "wb") as buffer:
            shutil.copyfileobj(multi_face_img.file, buffer)

        with open(single_path, "wb") as buffer:
            shutil.copyfileobj(single_face_img.file, buffer)

        logging.info("Uploaded files saved. Invoking face swap pipeline...")

        # Run the pipeline
        result_path = initiate_face_swapper(
            multi_face_img_path=multi_path,
            single_face_img_path=single_path
        )

        logging.info(f"Returning result image: {result_path}")
        return FileResponse(result_path, media_type="image/jpeg", filename=os.path.basename(result_path))

    except CustomException as ce:
        logging.error("CustomException during face swap API", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(ce)})

    except Exception as e:
        logging.error("Unhandled exception during face swap API", exc_info=True)
        raise CustomException(e, sys) from e

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)