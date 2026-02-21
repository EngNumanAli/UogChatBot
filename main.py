from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional
from dotenv import load_dotenv
import os

# ✅ FIX 1: Correct import — langchain_classic does NOT exist
from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# ✅ FIX 3: Load API key from .env file — never hardcode secrets
load_dotenv()

# ─── Global state ────────────────────────────────────────────────────────────
qa_chain = None


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    try:
        print("🔄 Loading AI system...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        vector_store = FAISS.load_local(
            "faiss_index_uog",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            # ✅ Reads GROQ_API_KEY from environment automatically
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        )
        print("✅ System loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading system: {e}")

    yield
    print("🛑 Shutting down...")


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(title="UOG Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ──────────────────────────────────────────────────────────────────

# ✅ FIX 2: Field names now match exactly what the frontend sends:
#    { "query": "...", "conversation_id": "..." }
#    No aliases, no capital letters, no camelCase mismatch.
class ChatRequest(BaseModel):
    query: Optional[str] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    result: str
    conversation_id: str
    success: bool


# ─── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "UOG Chatbot API is running", "status": "healthy"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    print(f"📩 query={request.query!r}  conv_id={request.conversation_id!r}")

    if qa_chain is None:
        raise HTTPException(status_code=503, detail="System is still loading. Please try again shortly.")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = qa_chain.invoke({"query": request.query.strip()})
        result   = response["result"]
        print(f"✅ Response: {result[:120]}...")

        return ChatResponse(
            result=result,
            conversation_id=request.conversation_id or "default",
            success=True,
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if qa_chain is not None else "initializing",
        "system_ready": qa_chain is not None,
    }


@app.post("/api/debug")
async def debug(request: Request):
    """Utility endpoint — logs exactly what JSON body was received."""
    body = await request.json()
    print(f"🔍 DEBUG body: {body}")
    return {"received": body}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)