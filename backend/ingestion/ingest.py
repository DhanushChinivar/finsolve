import os
import uuid
import logging
from typing import List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "finbot")

COLLECTION_ACCESS = {
    "general":     ["employee", "finance", "engineering", "marketing", "c_level"],
    "finance":     ["finance", "c_level"],
    "engineering": ["engineering", "c_level"],
    "marketing":   ["marketing", "c_level"],
    "hr":          ["c_level"],
}

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def create_collection():
    existing = client.get_collections()
    existing_names = [c.name for c in existing.collections]
    if COLLECTION_NAME not in existing_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        logger.info(f"Created collection: {COLLECTION_NAME}")
    else:
        logger.info(f"Collection already exists: {COLLECTION_NAME}")

def embed_and_store(chunks: List[Dict[str, Any]]):
    points = []
    for chunk in chunks:
        vector = embedder.encode(chunk["content"]).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "content": chunk["content"],
                "metadata": chunk.get("metadata", {})
            }
        )
        points.append(point)
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def load_documents(data_path: str) -> List[Dict[str, Any]]:
    all_chunks = []

    for collection in os.listdir(data_path):
        collection_path = os.path.join(data_path, collection)
        if not os.path.isdir(collection_path):
            continue

        for filename in os.listdir(collection_path):
            file_path = os.path.join(collection_path, filename)
            if not os.path.isfile(file_path):
                continue

            access_roles = COLLECTION_ACCESS.get(collection, [])
            ext = os.path.splitext(filename)[1].lower()

            if ext in (".pdf", ".md"):
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                result = converter.convert(file_path)
                text = result.document.export_to_markdown()
            elif ext == ".docx":
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([p.text for p in doc.paragraphs if p.text])
            elif ext == ".csv":
                import csv
                with open(file_path, "r") as f:
                    reader = csv.reader(f)
                    text = "\n".join([" | ".join(row) for row in reader])
            else:
                continue

            for chunk_text in split_into_chunks(text):
                chunk = {
                    "content": chunk_text,
                    "metadata": {
                        "source_document": filename,
                        "collection":      collection,
                        "access_roles":    access_roles,
                        "section_title":   None,
                        "page_number":     None,
                        "chunk_type":      "text",
                        "parent_chunk_id": None,
                    }
                }
                all_chunks.append(chunk)

    return all_chunks

if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "../data")
    create_collection()
    chunks = load_documents(data_path)
    embed_and_store(chunks)
    logger.info(f"Done! Ingested {len(chunks)} chunks total.")