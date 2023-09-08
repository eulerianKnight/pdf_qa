import streamlit as st
from pdf_qa import PdfQA
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import shutil

# Streamlit app code
st.set_page_config(
    page_title='Q&A ChatBot for PDF Files', 
    page_icon='📕', 
    layout='wide', 
    initial_sidebar_state='auto'
)

if "pdf_qa_model" not in st.session_state:
    st.session_state['pdf_qa_model']:PdfQA = PdfQA() # Initialize Model

# Cache resources for llm
@st.cache_resource
def load_llm():
    return PdfQA.create_llama_2_model()

# Cache resources for embedding model
@st.cache_resource
def load_embedding_model():
    return PdfQA.create_embeddings()

st.title("PDF Question Answering")


def main():
    with st.sidebar:
        pdf_file = st.file_uploader("**Upload PDF**", type="pdf")

        if st.button("Submit") and pdf_file is not None:
            with st.spinner(text="Uploading PDF and Generating Embeddings..."):
                with NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                    shutil.copyfileobj(pdf_file, tmp)
                    tmp_path = Path(tmp.name)
                    st.session_state['pdf_qa_model'].config = {
                        "pdf_path": str(tmp_path), 
                    }
                    st.session_state['pdf_qa_model'].embedding = load_embedding_model()
                    st.session_state['pdf_qa_model'].llm = load_llm()
                    st.session_state['pdf_qa_model'].init_embeddings()
                    st.session_state['pdf_qa_model'].init_models()
                    st.session_state['pdf_qa_model'].vector_db_pdf()
                    st.sidebar.success("PDF uploaded successfully")

    question = st.text_input('Ask a question', 'What is this document?')

    if st.button("Answer"):
        try:
            st.session_state['pdf_qa_model'].retrieval_qa_chain()
            answer = st.session_state['pdf_qa_model'].answer_query(question)
            st.write(f"{answer}")
        except Exception as e:
            st.error(f"Error answering the question: {str(e)}")

if __name__ == "__main__":
    main()