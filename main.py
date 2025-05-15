from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import boto3
import uuid
import time
import numpy as np
import faiss
from together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
from botocore.exceptions import ClientError
import os
import datetime
from boto3.dynamodb.types import Binary
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def time_operation(operation_name):
    """Context manager to time operations"""
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = (time.time() - start_time) * 1000  # milliseconds
        logger.info(f"TIMING: {operation_name} took {elapsed:.2f}ms")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- Secure Bucket Name Handling ---
def get_bucket():
    with time_operation("Bucket name retrieval"):
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId="rag-app/s3-bucket")
            return response['SecretString']
        except ClientError as e:
            logger.error(f"AWS Secrets Error: {e}")
            raise RuntimeError(f"AWS Secrets Error: {e}")

# --- Secure Table Name Handling ---
def get_table():
    with time_operation("Table name retrieval"):
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId="rag-app/dynamodb-table")
            return response['SecretString']
        except ClientError as e:
            logger.error(f"AWS Secrets Error: {e}")
            raise RuntimeError(f"AWS Secrets Error: {e}")
          
# AWS Clients
s3 = boto3.client('s3', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# AWS Resource Names
S3_BUCKET = get_bucket()
DYNAMO_TABLE = get_table()

# Initialize tables
try:
    with time_operation("DynamoDB table setup"):
        table = dynamodb.Table(DYNAMO_TABLE)
        table.load()
        logger.info("DynamoDB table loaded")
except Exception as e:
    logger.warning(f"Creating new DynamoDB table: {str(e)}")
    with time_operation("DynamoDB table creation"):
        table = dynamodb.create_table(
            TableName=DYNAMO_TABLE,
            KeySchema=[
                {'AttributeName': 'conversation_id', 'KeyType': 'HASH'},
                {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'conversation_id', 'AttributeType': 'S'},
                {'AttributeName': 'timestamp', 'AttributeType': 'N'}
            ],
            BillingMode='PAY_PER_REQUEST'
        )

# --- Secure API Key Handling ---
def get_api_key():
    with time_operation("API key retrieval"):
        try:
            client = boto3.client('secretsmanager')
            response = client.get_secret_value(SecretId="together-api-key")
            return response['SecretString']
        except ClientError as e:
            logger.error(f"AWS Secrets Error: {e}")
            raise RuntimeError(f"AWS Secrets Error: {e}")

try:
    with time_operation("Together client initialization"):
        together_client = Together(api_key=get_api_key())
except RuntimeError as e:
    logger.error(f"Failed to initialize Together client: {e}")
    raise

# --- RAG Components ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)

def process_document(file_bytes: bytes, file_extension: str):
    """Process document with timing"""
    with time_operation("Document processing"):
        if file_extension.lower() == '.pdf':
            from pypdf import PdfReader
            pdf_file = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_file)
            text = " ".join(page.extract_text() for page in reader.pages)
        else:
            text = file_bytes.decode('utf-8')
        return text_splitter.split_text(text)

def create_vector_index(chunks: list):
    """Create and serialize FAISS index with timing"""
    with time_operation("Embedding generation"):
        embeddings = together_client.embeddings.create(
            input=chunks,
            model="togethercomputer/m2-bert-80M-8k-retrieval"
        ).data
        embeddings_array = np.array([e.embedding for e in embeddings]).astype('float32')
    
    with time_operation("Index creation"):
        quantizer = faiss.IndexFlatIP(embeddings_array.shape[1])
        index = faiss.IndexIDMap(quantizer)
        index.add_with_ids(embeddings_array, np.arange(len(chunks)))
    
    with time_operation("Index serialization"):
        return faiss.serialize_index(index).tobytes()

def get_answer(question: str, index, chunks):
    """RAG pipeline execution with detailed timing"""
    timings = {}
    
    try:
        # Get question embedding
        with time_operation("Question embedding"):
            question_embedding = together_client.embeddings.create(
                input=question,
                model="togethercomputer/m2-bert-80M-8k-retrieval"
            ).data[0].embedding
        
        # Retrieve relevant chunks
        with time_operation("FAISS search"):
            scores, indices = index.search(np.array([question_embedding]).astype('float32'), k=5)
            context = "\n\n".join([chunks[i] for i in indices[0]])
        
        # Generate answer
        prompt = f"""Answer the question using only the provided context:
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:"""
        
        with time_operation("LLM generation"):
            response = together_client.completions.create(
                prompt=prompt,
                model="meta-llama/Llama-3-70b-chat-hf",
                max_tokens=512,
                temperature=0.3
            )
            answer = response.choices[0].text.strip()
        
        return answer
    except Exception as e:
        logger.error(f"Error in get_answer: {str(e)}", exc_info=True)
        raise RuntimeError(f"Failed to generate answer: {str(e)}")

# --- API Endpoints ---
@app.get("/")
async def chat_interface(request: Request):
    with time_operation("Chat interface render"):
        return templates.TemplateResponse("chat.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    total_start = time.time()
    timings = {}
    conversation_id = None
    s3_key = None
    
    try:
        # File validation
        with time_operation("File validation"):
            if not file.filename:
                raise ValueError("Empty filename")
            file_extension = os.path.splitext(file.filename)[1]
            if not file_extension:
                raise ValueError("No file extension detected")

        # Generate IDs
        with time_operation("ID generation"):
            conversation_id = str(uuid.uuid4())
            file_id = str(uuid.uuid4())

        # Read file content
        with time_operation("File reading"):
            file_bytes = await file.read()
            if len(file_bytes) == 0:
                raise ValueError("Empty file content")

        # S3 upload
        with time_operation("S3 upload"):
            s3_key = f"uploads/{file_id}{file_extension}"
            s3.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=file_bytes,
                ContentType=file.content_type,
                Metadata={
                    "original_filename": file.filename,
                    "conversation_id": conversation_id
                }
            )

        # S3 verification
        with time_operation("S3 verification"):
            s3_head = s3.head_object(Bucket=S3_BUCKET, Key=s3_key)
            if s3_head['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise ValueError("S3 HEAD request failed")

        # Document processing
        with time_operation("Document processing"):
            chunks = process_document(file_bytes, file_extension)

        # Vector index creation
        with time_operation("Vector index creation"):
            vector_index = create_vector_index(chunks)

        # DynamoDB storage
        with time_operation("DynamoDB write"):
            db_item = {
                'conversation_id': conversation_id,
                'timestamp': int(time.time()),
                'type': 'metadata',
                'filename': file.filename,
                's3_key': s3_key,
                'file_size': len(file_bytes),
                'content_type': file.content_type,
                'chunks': chunks,
                'chunk_count': len(chunks),
                'index': vector_index,
                'messages': [],
                's3_etag': s3_head['ETag']
            }
            
            db_response = table.put_item(Item=db_item)
            if db_response['ResponseMetadata']['HTTPStatusCode'] != 200:
                raise ValueError("DynamoDB write failed")

        total_time = (time.time() - total_start) * 1000
        logger.info(f"TOTAL UPLOAD TIME: {total_time:.2f}ms")
        
        return JSONResponse({
            "filename": db_item['filename'],
            "status": "success",
            "conversation_id": conversation_id,
            "s3_url": f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}",
            "timings": {
                "total": total_time,
                "stages": {
                    "file_processing": timings.get("Document processing", 0),
                    "embedding_generation": timings.get("Embedding generation", 0),
                    "index_creation": timings.get("Vector index creation", 0),
                    "storage": timings.get("DynamoDB write", 0) + timings.get("S3 upload", 0)
                }
            }
        })

    except Exception as e:
        logger.error(f"Upload failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))

@app.post("/ask")
async def ask_question(request: Request):
    total_start = time.time()
    timings = {}
    
    try:
        # Parse input
        with time_operation("Request parsing"):
            data = await request.json()
            print(data)
            conversation_id = data.get("conversation_id")
            question = data.get("question")
            
            if not conversation_id:
                raise HTTPException(400, detail="Missing conversation_id")
            if not question:
                raise HTTPException(400, detail="Missing question")

        # Get metadata item (most recent one)
        with time_operation("DynamoDB query"):
            # First get all items for this conversation
            response = table.query(
                KeyConditionExpression="conversation_id = :cid",
                ExpressionAttributeValues={":cid": str(conversation_id)},
                ScanIndexForward=False  # Newest items first
            )
            
            if not response.get('Items'):
                raise HTTPException(404, detail=f"Conversation {conversation_id} not found")
            
            # Find the metadata item (could be first or need to search)
            metadata_item = None
            for item in response['Items']:
                if item.get('type') == 'metadata':
                    metadata_item = item
                    break
            
            if not metadata_item:
                raise HTTPException(404, detail="No document metadata found")

        # Index handling
        with time_operation("Index deserialization"):
            index_data = metadata_item['index']
            if isinstance(index_data, Binary):
                index_bytes = index_data.value
            elif isinstance(index_data, (bytes, bytearray)):
                index_bytes = index_data
            elif isinstance(index_data, str):
                index_bytes = index_data.encode('latin-1')
            else:
                raise ValueError(f"Unexpected index type: {type(index_data)}")
            
            index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype='uint8'))

        # Get RAG answer
        with time_operation("RAG pipeline"):
            answer = get_answer(question, index, metadata_item['chunks'])

        # Update conversation history
        with time_operation("Conversation update"):
            update_response = table.put_item(Item={
                'conversation_id': str(conversation_id),
                'timestamp': int(time.time()),
                'type': 'message',
                'question': question,
                'answer': answer
            })

        total_time = (time.time() - total_start) * 1000
        logger.info(f"TOTAL ASK TIME: {total_time:.2f}ms")
        
        return JSONResponse({
            "answer": answer,
            "conversation_id": conversation_id,
            "timings": {
                "total": total_time,
                "stages": {
                    "dynamodb_query": timings.get("DynamoDB query", 0),
                    "index_processing": timings.get("Index deserialization", 0),
                    "rag_pipeline": timings.get("RAG pipeline", 0)
                }
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask failed: {str(e)}", exc_info=True)
        raise HTTPException(500, detail=str(e))