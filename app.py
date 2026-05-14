import streamlit as st
import os
from pathlib import Path
import google.generativeai as genai

# --- Configuration ---
st.set_page_config(page_title="Vault-X: Document Q&A", page_icon="📚")
st.title("📚 Vault-X: Document Q&A (Simplified)")
st.markdown("Ask questions about your documents. No complex database needed.")

# Get API Key from Streamlit Secrets
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Please set the GEMINI_API_KEY in your Streamlit secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- CHANGED: Use gemini-2.5-flash instead of 2.0 ---
model = genai.GenerativeModel('gemini-2.5-flash')

# --- Load Documents ---
def load_documents():
    docs_dir = Path("documents")
    if not docs_dir.exists():
        return None, "No 'documents' folder found."

    all_text = []
    for file_path in docs_dir.iterdir():
        if file_path.is_file():
            try:
                if file_path.suffix.lower() == ".pdf":
                    from pypdf import PdfReader
                    reader = PdfReader(str(file_path))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()
                    all_text.append(text)
                    st.info(f"Loaded PDF: {file_path.name}")
                elif file_path.suffix.lower() == ".txt":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    all_text.append(text)
                    st.info(f"Loaded TXT: {file_path.name}")
            except Exception as e:
                st.warning(f"Could not load {file_path.name}: {e}")
    
    if not all_text:
        return None, "No documents found."
    
    return "\n\n".join(all_text), f"Loaded {len(all_text)} document(s)."

# --- Load documents into session state ---
if "documents_content" not in st.session_state:
    with st.spinner("Loading documents..."):
        content, message = load_documents()
        st.session_state.documents_content = content
        st.session_state.documents_loaded = content is not None
        if content:
            st.success(message)
        else:
            st.error(message)

# --- Query Section ---
if st.session_state.documents_loaded:
    st.subheader("Ask a question about your documents")
    query = st.text_input("Your question:", placeholder="e.g., What is the vacation policy?")
    
    if query:
        with st.spinner("Analyzing documents and generating answer..."):
            try:
                prompt = f"""
                Based on the following document content, answer the question.
                
                DOCUMENT CONTENT:
                {st.session_state.documents_content[:15000]}
                
                QUESTION: {query}
                
                Answer concisely and accurately. If the answer cannot be found in the document, say "I cannot find this information in the provided documents."
                """
                
                response = model.generate_content(prompt)
                
                st.subheader("📝 Answer")
                st.write(response.text)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("📁 Add PDF or TXT files to the 'documents' folder in your GitHub repository and click Rerun.")

# Sidebar
with st.sidebar:
    st.markdown("### 🔧 System Details")
    st.markdown("""
    - **Model**: Gemini 2.5 Flash
    - **Document Loading**: Direct (no vector DB)
    - **Supported files**: PDF, TXT
    """)
