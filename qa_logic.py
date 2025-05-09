
import numpy as np
from together import Together
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os 

load_dotenv()  # Load environment variables from .env file

# Initialize Together client (replace with your API key)
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# Load your document (e.g., PDF, TXT)
def load_document(file_path):
    if file_path.endswith('.pdf'):
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        return " ".join(page.extract_text() for page in reader.pages)
    else:
        with open(file_path, 'r') as f:
            return f.read()

document = load_document("sample.pdf")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust based on your needs
    chunk_overlap=100
)

chunks = text_splitter.split_text(load_document("sample.pdf"))

# Get embeddings for all chunks
def get_embeddings(texts):
    return [client.embeddings.create(
        input=text,
        model="togethercomputer/m2-bert-80M-8k-retrieval"
    ).data[0].embedding for text in texts]

chunk_embeddings = get_embeddings(chunks)

import faiss
import numpy as np

# Convert embeddings to numpy array
embeddings_array = np.array(chunk_embeddings).astype('float32')
print(embeddings_array.shape)

# Create and save index
index = faiss.IndexFlatIP(embeddings_array.shape[1])  # Inner product similarity
index.add(embeddings_array)
faiss.write_index(index, "my_index.faiss")

def embed_query(query):
    response = client.embeddings.create(
        input=query,
        model="togethercomputer/m2-bert-80M-8k-retrieval"
    )
    return np.array(response.data[0].embedding, dtype='float32')

def retrieve(query, top_k=1):
    query_embedding = embed_query(query)
    query_embedding = np.expand_dims(query_embedding, axis=0)  # shape (1, embedding_dim)
    
    scores, indices = index.search(query_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    
    return results

def answer_query(query, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"Answer the following question using the provided context:\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    
    response = client.completions.create(
        prompt=prompt,
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        max_tokens=512,
        temperature=0.1
    )
    return response.choices[0].text.strip()

retrieved = retrieve("What are the main findings?")
# print(retrieved)
answer = answer_query("What are the main findings?", retrieved)
print(answer)
