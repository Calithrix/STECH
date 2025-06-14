import os
import streamlit as st
import torch
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer


# Load environment variables
load_dotenv(r"C:/Users/Matthew/Dropbox/The App/pdf_parser/.env")
XAI_API_KEY=os.getenv("XAI_API_KEY")
print(f"Using API Key: {xai_api_key}")
if not xai_api_key:
    st.error("XAI_API_KEY not set. Please check .env or Streamlit secrets.")
    st.stop()

# Initialize session state for chat history and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Streamlit app layout
st.set_page_config(page_title="PDF Parser with Grok", page_icon="📄")
st.title("📄 PDF Parser with Grok")
st.write("Upload a PDF and ask questions about its content!")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process the PDF
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    
    # Initialize embeddings with explicit device setting
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Manually initialize SentenceTransformer to avoid meta tensor issue
embeddings.client = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device="cpu"
)
print(f"Embeddings device: {embeddings.client.device}")  # Debug
    # Force CPU to avoid meta tensor issue
print(f"Embeddings initialized on device: {embeddings.client.device}")  # Debug
    
    # Create vector store
st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Initialize Grok LLM
llm = ChatGroq(
        model_name="llama-3.1-70b-versatile",
        api_key=xai_api_key,
        temperature=0.7
    )
    
    # Define prompt template
prompt_template = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template="""
        You are a helpful assistant that answers questions based on the provided PDF content.
        Use the following context to answer the question. If the answer is not in the context,
        say so and provide a general response if applicable.

        Chat History: {chat_history}
        Context: {context}
        Question: {question}

        Answer:
        """
    )
    
    # Create conversational retrieval chain
st.session_state.conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    
st.success("PDF processed successfully! You can now ask questions.")

# validation check for API keys 
print(f"API Key: {XAI_API_KEY}")
if not XAI_API_KEY:
    st.error("XAI_API_KEY is not set. Please check .env or Streamlit secrets.")
    st.stop()
    
# Chat interface
st.subheader("Ask a Question")
user_question = st.text_input("Enter your question about the PDF:", key="question")

if user_question and st.session_state.conversation:
    # Get response from the conversation chain
    result = st.session_state.conversation({
        "question": user_question,
        "chat_history": st.session_state.chat_history
    })
    
    # Update chat history
    st.session_state.chat_history.append((user_question, result["answer"]))
    
    # Display the answer
    st.write("**Answer:**")
    st.write(result["answer"])
    
    # Display chat history
    st.subheader("Chat History")
    for question, answer in st.session_state.chat_history:
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {answer}")
        st.write("---")

# Clean up temporary file
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")