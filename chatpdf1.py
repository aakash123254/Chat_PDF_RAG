import os
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# ✅ Load .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing. Check your .env file.")
    st.stop()

# ✅ Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ✅ PDF Processing Function
def process_pdf(pdf_path):
    """Extracts text from a PDF and creates a FAISS vector store."""
    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    # ✅ Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # ✅ Create FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ✅ Streamlit UI
st.title("📄 PDF Chatbot using Google Gemini")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_path = f"./pdfs/{uploaded_file.name}"
    
    # Save uploaded file
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")
    vector_store = process_pdf(pdf_path)
    st.session_state["vector_store"] = vector_store

# ✅ User Query Section
query = st.text_input("Ask a question about the PDF:")
if query and "vector_store" in st.session_state:
    vector_store = st.session_state["vector_store"]

    # ✅ Search for relevant document sections
    docs = vector_store.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in docs])

    # ✅ Generate Answer
    response = f"**Relevant Sections:**\n{context}\n\n**Answer:**\n(TODO: Implement response generation)"
    
    st.write(response)
