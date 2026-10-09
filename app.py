import os
import requests
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

st.set_page_config(page_title="Traffic RAG Assistant", page_icon="🚦", layout="wide")
KB_DIR = "knowledge_base"
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"


st.markdown("""
<style>
.main-title {font-size:38px;font-weight:700;margin-bottom:5px;}
.subtitle {color:#666;margin-bottom:25px;}
.chat-user {background:#eef4ff;padding:12px 16px;border-radius:12px;margin:8px 0;}
.chat-assistant {background:#f3f3f3;padding:12px 16px;border-radius:12px;margin:8px 0 16px 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_documents():
    documents = []
    if not os.path.exists(KB_DIR):
        return documents
    for filename in sorted(os.listdir(KB_DIR)):
        path = os.path.join(KB_DIR, filename)
        if filename.lower().endswith(".pdf"):
            try:
                reader = PdfReader(path)
                text = "\n".join((page.extract_text() or "") for page in reader.pages)
                if text.strip(): documents.append({"source": filename, "text": text})
            except Exception as e:
                st.warning(f"Could not read {filename}: {e}")
        elif filename.lower().endswith(".txt"):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f: text = f.read()
                if text.strip(): documents.append({"source": filename, "text": text})
            except Exception as e:
                st.warning(f"Could not read {filename}: {e}")
    return documents

def make_chunks(documents, chunk_words=700, overlap=100):
    chunks = []
    for doc in documents:
        words = doc["text"].split()
        start = 0
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunks.append({"source": doc["source"], "text": " ".join(words[start:end])})
            if end == len(words): break
            start = end - overlap
    return chunks

@st.cache_resource(show_spinner="Loading AI embedding model...")
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner="Building knowledge-base index...")
def build_index():
    documents = load_documents()
    if not documents: return None, [], None
    chunks = make_chunks(documents)
    model = get_embedding_model()
    embeddings = np.asarray(model.encode([c["text"] for c in chunks], normalize_embeddings=True), dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, chunks, model

TRAFFIC_KEYWORDS = {
    "traffic","road","vehicle","car","bike","motorcycle","speed","speeding","speed limit",
    "fine","penalty","violation","challan","license","licence","driving","driver",
    "motor vehicle","overtaking","parking","registration","tax","token","traffic rule",
    "traffic law","section","ordinance","motor vehicles rules"
}
GREETING_WORDS = {"hi","hello","hey","salam","assalamualaikum","good morning","good afternoon","good evening"}

def is_greeting(q):
    return q.lower().strip().replace("!","").replace("?","") in GREETING_WORDS

def is_general_conversation(q):
    q = q.lower().strip()
    return any(p in q for p in ["how are you","who are you","what are you","what can you do","your name","thank you","thanks","good morning","good afternoon","good evening"])

def is_traffic_question(q):
    q = q.lower()
    return any(k in q for k in TRAFFIC_KEYWORDS)

def ask_ollama(prompt):
    try:
        r = requests.post(OLLAMA_URL, json={"model":OLLAMA_MODEL,"prompt":prompt,"stream":False}, timeout=120)
        if r.status_code != 200:
            return f"Ollama returned HTTP {r.status_code}. Please make sure Ollama is running and '{OLLAMA_MODEL}' is installed."
        return r.json().get("response", "").strip() or "I could not generate an answer."
    except requests.exceptions.ConnectionError:
        return "I cannot connect to Ollama. Open Command Prompt and run:\n\n`ollama run llama3.2`\n\nThen try again."
    except requests.exceptions.Timeout:
        return "The AI model took too long to respond. Please try again."
    except Exception as e:
        return f"An error occurred while contacting the AI model: {e}"

def retrieve(question, index, chunks, model, top_k=4):
    if index is None or not chunks: return []
    q = np.asarray(model.encode([question], normalize_embeddings=True), dtype="float32")
    scores, ids = index.search(q, min(top_k, len(chunks)))
    return [{"source":chunks[i]["source"],"text":chunks[i]["text"],"score":float(s)} for s,i in zip(scores[0],ids[0]) if i >= 0]

def answer_traffic_question(question, retrieved):
    if not retrieved:
        return "I could not find relevant information in the traffic-law knowledge base. Please add the relevant official document."
    context = "\n\n---\n\n".join(f"Source: {x['source']}\n{x['text']}" for x in retrieved)
    prompt = f"""You are a Traffic RAG Assistant. Answer the user's question using ONLY the supplied traffic-law context. Do not invent speed limits, fines, penalties, sections, or legal requirements. If the context does not contain the answer, say so clearly. Give a short, clear answer suitable for a student project demonstration and mention the source when useful.\n\nTraffic-law context:\n{context}\n\nUser question:\n{question}\n\nAnswer:"""
    return ask_ollama(prompt)

documents = load_documents()
with st.sidebar:
    st.markdown("## 📚 Knowledge Base")
    st.write(f"Documents loaded: **{len(documents)}**")
    for doc in documents: st.write(f"📄 {doc['source']}")
    if not documents: st.warning("No PDF/TXT documents found in knowledge_base.")
    st.markdown("---")
    st.markdown("### Example questions")
    st.write("• What is the speed limit?")
    st.write("• What is speeding?")
    st.write("• What penalty applies?")
    st.write("• Explain this traffic rule.")

st.markdown('<div class="main-title">🚦 Traffic RAG Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask general questions or questions about traffic rules stored in the knowledge base.</div>', unsafe_allow_html=True)

index = chunks = model = None
if documents:
    try: index, chunks, model = build_index()
    except Exception as e: st.error(f"Knowledge-base index could not be built: {e}")

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'<div class="chat-user">👤 <b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-assistant">🤖 <b>Assistant:</b><br>{message["content"]}</div>', unsafe_allow_html=True)

question = st.chat_input("Ask a question...")
if question:
    question = question.strip()
    if not question: st.stop()
    st.session_state.messages.append({"role":"user","content":question})
    if is_greeting(question):
        answer = "Hello! 👋 I am your Traffic RAG Assistant. How can I help you?"
    elif is_general_conversation(question):
        q = question.lower()
        if "how are you" in q: answer = "I'm doing great! 😊 I'm ready to help you with traffic rules and regulations."
        elif "who are you" in q or "what are you" in q: answer = "I am a Traffic RAG Assistant that uses traffic-law documents to answer traffic-related questions."
        elif "what can you do" in q: answer = "I can answer traffic-rule questions using the documents in my knowledge base and explain the relevant information."
        elif "thank" in q: answer = "You're welcome! 😊"
        else: answer = "Sure! How can I help you with traffic rules?"
    elif is_traffic_question(question):
        with st.spinner("🔎 Searching traffic rules..."):
            retrieved = retrieve(question, index, chunks, model, top_k=4)
            answer = answer_traffic_question(question, retrieved)
    else:
        answer = "I can answer general greetings and traffic-related questions. For example: **What is speeding?** or **What is the speed limit?**"
    st.session_state.messages.append({"role":"assistant","content":answer})
    st.rerun()
