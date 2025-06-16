import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# load environment variables
load_dotenv(r"C:/Users/Matthew/Dropbox/The App/pdf_parser/.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# initialize session state for chat history and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation" not in st.session_state:
    st.session_state.conversation = None

# Streamlit app layout
st.set_page_config(page_title="STECH PDF Parser", page_icon="📄")
st.title("📄 STECH PDF Parser")
st.write("Upload a PDF and ask questions about its content or pull direct quotes!")

# upload file 
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")



if uploaded_file is not None:
    # save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    # load and process PDF
    loader = PyPDFLoader("temp.pdf", extract_images=False)
    documents = loader.load()
    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    # initialize embeddings with explicit device setting
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)

    # Create vector store
st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Initialize chatgpt 
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # Define prompt template
prompt_template = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template="""..."""
)
    # Create conversational retrieval chain
qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True
    )
    
st.success("PDF processed successfully! You can now ask questions.")

# validation check for API keys 
print(f"API Key: {OPENAI_API_KEY}")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please check .env or Streamlit secrets.")
    st.stop()
    
# Chat interface
st.subheader("Ask a Question")
user_question = st.text_input("Enter your question about the PDF:", key="question")

if user_question and st.session_state.conversation:
    # Get response from the conversation chain
    result = st.session_state.qa_chain({"query": user_question})
    answer = result["result"]
    sources = result["source_documents"]
    
    # Update chat history
    st.session_state.chat_history.append((user_question, answer))
    
    # Display the answer
    st.write("**Answer**: ", answer)
    st.write("**Sources**:")
    for doc in sources:
        page = doc.metadata.get("page", "N/A")
        content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        st.write(f"- \"{content}\" (Page {page})")

    # Display chat history
    st.subheader("Chat History")
    for question, answer in st.session_state.chat_history:
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {answer}")
        st.write("---")

# Clean up temporary file
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")