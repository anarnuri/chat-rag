from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from pathlib import Path
from typing import List
import time

# Initialize app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Store conversation history (in-memory for this example)
conversations = {}

@app.get("/")
async def chat_interface(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Generate unique filename
        file_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1]
        file_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
        
        # Save file
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                buffer.write(chunk)
        
        # Initialize conversation
        conversations[file_id] = {
            "filename": file.filename,
            "path": file_path,
            "messages": []
        }
        
        return JSONResponse({
            "file_id": file_id,
            "filename": file.filename,
            "status": "success"
        })
    
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    file_id = data.get("file_id")
    question = data.get("question")
    
    if file_id not in conversations:
        raise HTTPException(404, detail="File not found")
    
    # Add user question to history
    conversations[file_id]["messages"].append({
        "sender": "user",
        "text": question,
        "timestamp": time.time()
    })
    
    # Process question (replace with your actual QA logic)
    answer = "This is a simulated answer. Implement your QA logic here."
    
    # Add system response to history
    conversations[file_id]["messages"].append({
        "sender": "assistant",
        "text": answer,
        "timestamp": time.time()
    })
    
    return JSONResponse({
        "answer": answer,
        "conversation": conversations[file_id]["messages"]
    })