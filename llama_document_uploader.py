import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.groq import Groq
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

# Initialize models
llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

embedding_model = OllamaEmbedding(model_name="nomic-embed-text")

# UI
st.title("Q&A with Your Documents")
st.markdown("Upload your `.pdf` files to build a semantic search index and ask questions.")

uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
query = st.text_input("Ask a question about the documents")

if uploaded_files and query:
    with st.spinner("Processing documents and building index..."):
        with tempfile.TemporaryDirectory() as temp_dir:

            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

            # Load documents
            documents = SimpleDirectoryReader(input_dir=temp_dir).load_data()

            # Build index
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=embedding_model
            )

            # Query engine
            query_engine = index.as_query_engine(llm=llm)
            response = query_engine.query(query)

            st.subheader("Answer:")
            st.write(response.response)