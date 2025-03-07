from fastapi import FastAPI, File, UploadFile, Query
import os
import shutil
from functionlity import *

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

UPLOAD_DIR = "data/input_videos/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    """Upload video file and save it."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"message": "Video uploaded successfully", "filename": file.filename, "file_path": file_path}

@app.get("/transcribe/")
async def transcribe_video(file_path: str = Query(..., description="Path of the uploaded video file")):
    """Transcribe the uploaded video file."""
    chunk_data = transcript_audio(file_path)
    faiss(chunk_data)
    return {"transcription": chunk_data}

@app.get("/answer/")
def get_answer(question: str = Query(..., description="User's question")):
    """Answer the question based on the context."""
    response = answer_question(question)  # Fixed function call
    return {"question": question, "answer": response}
