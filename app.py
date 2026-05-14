import streamlit as st
import os
from pathlib import Path

# CORRECTED IMPORTS for newer LangChain versions
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Changed
from langchain.chains import RetrievalQA

# For markdown files (optional)
try:
    from langchain_community.document_loaders import UnstructuredMarkdownLoader
except ImportError:
    UnstructuredMarkdownLoader = None
    st.warning("Markdown loader not available. Only PDF and TXT files will be supported.")

# --- Configuration ---
st.set_page_config(page_title="Vault-X: Private Knowledge Base", page_icon="📚")
st.title("📚 Vault-X: Private Document Q&A")
st.markdown("Ask questions about your private documents. Your data stays in your control.")

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
    
    # 1. Load PDF and TXT files
    for file_path in docs_dir.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                    all_docs.extend(loader.load())
                    st.info(f"Loaded PDF: {file_path.name}")
                elif file_path.suffix.lower() == ".txt":
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    all_docs.extend(loader.load())
                    st.info(f"Loaded TXT: {file_path.name}")
                elif file_path.suffix.lower() == ".md" and UnstructuredMarkdownLoader:
                    loader = UnstructuredMarkdownLoader(str(file_path))
                    all_docs.extend(loader.load())
                    st.info(f"Loaded MD: {file_path.name}")
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
    
    st.info(f"Split into {len(chunks)} chunks for indexing.")

    # 3. Use Gemini Embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    # 4. Store in ChromaDB (Persistent)
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    st.success("Vector database created successfully!")
    return vector_db

# --- Main Logic ---

# Initialize or load the database
vector_db = initialize_vector_db()

if vector_db:
    # Initialize the LLM (Gemini 2.5 Flash)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.3
    )

    # Setup the Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )

    # User Query Input
    st.subheader("Ask a question about your documents")
    query = st.text_input("Your question:", placeholder="e.g., What is the vacation policy?")

    if query:
        with st.spinner("Searching documents and generating answer..."):
            try:
                response = qa_chain.invoke({"query": query})
                
                # Display Answer
                st.subheader("📝 Answer")
                st.markdown(response["result"])
                
                # Show Source Documents
                if response.get("source_documents"):
                    st.subheader("📎 Sources")
                    for i, doc in enumerate(response["source_documents"][:3]):  # Limit to 3 sources
                        source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
                        with st.expander(f"Source {i+1}: {source_name}"):
                            # Show first 500 characters
                            content = doc.page_content[:500]
                            if len(doc.page_content) > 500:
                                content += "..."
                            st.write(content)
                            if "page" in doc.metadata:
                                st.caption(f"Page: {doc.metadata['page']}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("📁 Add PDF or TXT documents to the 'documents' folder in your GitHub repository and click Rerun.")

# Sidebar info
with st.sidebar:
    st.markdown("### 🔧 System Details")
    st.markdown("""
    - **Model**: Gemini 2.5 Flash
    - **Embeddings**: gemini-embedding-001
    - **Vector Store**: ChromaDB
    - **Supported files**: PDF, TXT, MD
    """)
    
    st.markdown("### 📖 How to use")
    st.markdown("""
    1. Upload PDF/TXT files to the `documents` folder in GitHub
    2. The app will automatically index them
    3. Ask any question about your documents
    4. Get AI-powered answers with citations
    """)
