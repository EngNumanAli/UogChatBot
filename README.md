🎓 UOG AI Assistant - RAG Chatbot
An intelligent, context-aware chatbot designed to assist students and applicants with queries regarding the University of Gujrat (UOG). Built using a Retrieval-Augmented Generation (RAG) architecture, this assistant leverages data scraped directly from the official UOG website to provide accurate, specific answers about admissions, fee structures, and campus deadlines.

✨ Features
Accurate Contextual Answers: Uses RAG to ground the AI's responses in factual university data, significantly reducing hallucinations.

Semantic Search: Employs FAISS (Facebook AI Similarity Search) for blazing-fast vector retrieval of relevant document chunks.

Open-Source LLM Powered: Integrates Groq API, utilizing state-of-the-art open-source models Ilama 3 for natural language generation.

Interactive UI: Features a clean, chat-like web interface built with Streamlit.


🛠️ Tech Stack
Language: Python

Framework: LangChain

Frontend UI: Streamlit

Vector Database: FAISS (Local)

Embeddings: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

LLM: Groq

🚀 Installation & Setup
1. Clone the repository

Bash
git clone https://github.com/EngNumanAli/uog-rag-chatbot.git
cd uog-rag-chatbot
2. Create a Virtual Environment (Recommended)

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies

Bash
pip install -r requirements.txt
(Ensure your requirements.txt includes langchain, langchain-huggingface, langchain-community, faiss-cpu, streamlit, huggingface_hub, and sentence-transformers)

4. Set up Grpq API Key
You need a free Groq Api  to get answer from the LLM.

Generate a  Api key from Groq console

Add it directly to the environment variables in your code, or create a .env file.

5. Prepare the Vector Database
Before running the chatbot, ensure your scraped UOG data has been embedded.

Run your ingestion script to generate the faiss_index_uog folder.

6. Run the Application

Bash
streamlit run app.py
🧠 How It Works
Data Ingestion: Raw text scraped from UOG web pages is split into smaller, manageable chunks.

Embedding: These chunks are converted into vector representations using a SentenceTransformer model and stored locally in a FAISS index.

Retrieval: When a user asks a question, the query is vectorized, and FAISS fetches the top 3 most semantically similar document chunks.

Generation: The retrieved context and the user's original question are passed into a custom prompt template and sent to the LLM to generate a helpful, targeted response.

👤 Author
Numan Ali
BS Software Engineering, University of Gujrat
