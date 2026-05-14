import streamlit as st
import os
from pathlib import Path

# LangChain and Gemini imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# --- Configuration ---
st.set_page_config(page_title="Gemini Knowledge Base", page_icon="📚")
st.title("📚 Local Knowledge Base (RAG)")
st.markdown("Query your local `.pdf`, `.txt`, and `.md` documents using Gemini 2.5 Flash.")

# 1. Get API Key from Streamlit Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please set the GEMINI_API_KEY in your Streamlit secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# --- Core Functions ---

@st.cache_resource(show_spinner="Processing documents and building vector database...")
def initialize_vector_db():
    """Reads documents, splits them, and stores them in a persistent ChromaDB."""
    docs_dir = Path("documents")
    if not docs_dir.exists():
        st.error("The 'documents' directory does not exist in the repository.")
        return None

    all_docs = []
    
    # 1. Load PDF, TXT, and MD files
    for file_path in docs_dir.iterdir():
        try:
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                all_docs.extend(loader.load())
            elif file_path.suffix == ".txt":
                loader = TextLoader(str(file_path))
                all_docs.extend(loader.load())
            elif file_path.suffix == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
                all_docs.extend(loader.load())
        except Exception as e:
            st.warning(f"Could not load {file_path.name}: {e}")

    if not all_docs:
        st.error("No valid documents found in the 'documents' directory.")
        return None

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(all_docs)

    # 3. Use Gemini Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Store in ChromaDB (Persistent)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vector_db

# --- Main Logic ---

# Initialize or load the database
vector_db = initialize_vector_db()

if vector_db:
    # Initialize the LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3,
        convert_system_message_to_human=True
    )

    # Setup the Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True # Required for citations
    )

    # User Query Input
    query = st.text_input("Ask a question about your documents:", placeholder="e.g., What is the main conclusion of the report?")

    if query:
        with st.spinner("Thinking..."):
            response = qa_chain.invoke({"query": query})
            
            # Display Answer
            st.subheader("Answer")
            st.write(response["result"])

            # 8. Show Source Documents/Citations
            st.subheader("Sources & Citations")
            for i, doc in enumerate(response["source_documents"]):
                with st.expander(f"Source {i+1}: {os.path.basename(doc.metadata.get('source', 'Unknown'))}"):
                    st.write(doc.page_content)
                    if "page" in doc.metadata:
                        st.caption(f"Page: {doc.metadata['page'] + 1}")
else:
    st.info("Add documents to the 'documents' folder and refresh the app to get started.")

# Sidebar info
with st.sidebar:
    st.info("""
    **System Details:**
    - **Model**: Gemini 2.5 Flash
    - **Embeddings**: gemini-embedding-001
    - **Vector Store**: ChromaDB (Persistent)
    - **Files**: PDF, TXT, MD
    """)
