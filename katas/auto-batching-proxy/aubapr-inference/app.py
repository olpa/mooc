import asyncio
import time
import zlib
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AUBAPR Inference Service", version="1.0.0")

# Global lock to ensure sequential processing of requests
request_lock = asyncio.Lock()

class EmbedRequest(BaseModel):
    inputs: List[str]

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]

def calculate_embedding(text: str) -> List[float]:
    """Calculate a 4-dimensional embedding vector from text using CRC32."""
    crc32_hash = zlib.crc32(text.encode('utf-8'))
    
    # Convert 4-byte CRC32 to 4 float values
    byte1 = (crc32_hash >> 24) & 0xFF
    byte2 = (crc32_hash >> 16) & 0xFF
    byte3 = (crc32_hash >> 8) & 0xFF
    byte4 = crc32_hash & 0xFF
    
    # Normalize to [-1, 1] range
    embedding = [
        (byte1 - 127.5) / 127.5,
        (byte2 - 127.5) / 127.5,
        (byte3 - 127.5) / 127.5,
        (byte4 - 127.5) / 127.5
    ]
    
    return embedding

@app.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    """Generate embeddings for input texts with 1 second delay per request (sequential processing)."""
    # Use lock to ensure requests are processed one by one
    async with request_lock:
        # Add 1 second delay per request (regardless of number of inputs)
        time.sleep(1)
        
        embeddings = []
        for text in request.inputs:
            # Calculate embedding vector (no additional delay)
            embedding = calculate_embedding(text)
            embeddings.append(embedding)
        
        return EmbedResponse(embeddings=embeddings)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)