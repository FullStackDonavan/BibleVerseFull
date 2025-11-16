from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import weaviate
import os

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Connect to Weaviate
client = weaviate.Client(WEAVIATE_URL)

# Ensure schema exists
schema = {
    "classes": [
        {
            "class": "Document",
            "vectorizer": "text2vec-transformers",
            "properties": [
                {"name": "title", "dataType": ["text"]},
                {"name": "content", "dataType": ["text"]}
            ]
        }
    ]
}
existing_classes = [cls["class"] for cls in client.schema.get()["classes"]]
if "Document" not in existing_classes:
    client.schema.create(schema)

# FastAPI app
app = FastAPI(title="Weaviate API", version="1.0")

class DocumentIn(BaseModel):
    title: str
    content: str

class SearchQuery(BaseModel):
    query: str
    limit: int = 5

@app.post("/add")
def add_document(doc: DocumentIn):
    try:
        client.data_object.create(
            {"title": doc.title, "content": doc.content},
            "Document"
        )
        return {"status": "success", "message": "Document added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
def search_documents(q: SearchQuery):
    try:
        result = client.query.get("Document", ["title", "content"]).with_near_text({"concepts": [q.query]}).with_limit(q.limit).do()
        return result["data"]["Get"]["Document"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
