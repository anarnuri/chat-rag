from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import uuid
from pathlib import Path
import time
import numpy as np
import faiss
from together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configuration
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- API Key Setup ---
def get_api_key():
    """Read API key from api.txt file"""
    try:
        key_path = Path(__file__).parent / "api.txt"
        with open(key_path, "r") as f:
            return f.read().strip()
    except Exception as e:
        raise RuntimeError(f"Failed to load API key: {str(e)}")

try:
    together_client = Together(api_key=get_api_key())
except Exception as e:
    raise RuntimeError(f"Failed to initialize Together client: {str(e)}")

# --- RAG Components ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

def process_document(file_path: str):
    """Extract and chunk document text"""
    if file_path.endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        text = " ".join(page.extract_text() for page in reader.pages)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    return text_splitter.split_text(text)

def create_vector_index(chunks: list):
    """Create FAISS index from document chunks"""
    try:
        embeddings = together_client.embeddings.create(
            input=chunks,
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        ).data
        embeddings_array = np.array([e.embedding for e in embeddings]).astype('float32')
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        index.add(embeddings_array)
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to create vector index: {str(e)}")

def get_answer(question: str, index, chunks):
    """RAG pipeline execution"""
    try:
        # Get question embedding
        question_embedding = together_client.embeddings.create(
            input=question,
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        ).data[0].embedding
        
        # Retrieve relevant chunks
        scores, indices = index.search(np.array([question_embedding]).astype('float32'), k=2)
        context = "\n\n".join([chunks[i] for i in indices[0]])
        
        # Generate answer
        prompt = f"""Answer the question using only the provided context:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        response = together_client.completions.create(
            prompt=prompt,
            model="meta-llama/Llama-3-70b-chat-hf",
            max_tokens=512,
            temperature=0.1
        )
        return response.choices[0].text.strip()
    except Exception as e:
        raise RuntimeError(f"Failed to generate answer: {str(e)}")

# --- API Endpoints ---
conversations = {}  # Stores all document conversations

@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

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
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
        
        # Process document and create vector store
        chunks = process_document(file_path)
        index = create_vector_index(chunks)
        
        # Initialize conversation
        conversations[file_id] = {
            "filename": file.filename,
            "path": file_path,
            "chunks": chunks,
            "index": index,
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
    
    # Get RAG answer
    try:
        answer = get_answer(
            question=question,
            index=conversations[file_id]["index"],
            chunks=conversations[file_id]["chunks"]
        )
    except Exception as e:
        answer = f"Error: {str(e)}"
    
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