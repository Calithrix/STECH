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
print(f"API Key loaded: {OPENAI_API_KEY[:5]}...")  # Debug (partial key)
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please check .env or Streamlit secrets.")
    st.stop()

# initialize session state for chat history and vector store
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# Streamlit app layout
st.set_page_config(page_title="STECH PDF Parser", page_icon="📄")
st.title("📄 STECH PDF Parser")
st.write("Upload a PDF and ask questions about its content or pull direct quotes!")

# upload file 
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

#UPLOAD FILE BLOCK 

if uploaded_file is not None:
    # save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # load and process PDF
    loader = PyPDFLoader("temp.pdf", extract_images=False)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents") # DEBUG

    # split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks")  # DEBUG    

    # initialize embeddings with explicit device setting
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vector store
    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vector store created")  # Debug

    # Initialize chatgpt 
    try:
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            api_key=OPENAI_API_KEY,
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        st.stop()
    
    # Define prompt template
    prompt_template = PromptTemplate(
        input_variables=["question", "context", "chat_history"],
        template="""
        You are a helpful assistant that answers questions based on the provided PDF content.
        Provide exact quotes when possible, including page numbers from the source.
        If the answer is not in the context, say so.
        
        Context: {context}
        Question: {query}

        Answer:
        """
    )
    # Create conversational retrieval chain
    st.session_state.qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )   
    print("QA chain created")  # Debug
    st.success("PDF processed successfully! You can now ask questions.")
else:
    st.info("Please upload a PDF to start asking questions.")

# validation check for API keys 
print(f"API Key: {OPENAI_API_KEY[:5]}")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is not set. Please check .env or Streamlit secrets.")
    st.stop()
    
# Chat interface
st.subheader("Ask a Question")
user_question = st.text_input("Enter your question about the PDF:", key="question")
submit_button = st.button("Submit Question")

print(f"User question: '{user_question}'")  # Debug
print(f"QA chain exists: {st.session_state.qa_chain is not None}")  # Debug

# CHAT INTERFACE

if submit_button and user_question and st.session_state.qa_chain:
    print("Processing query")  # Debug
    try:
        # Get response from the conversation chain
        result = st.session_state.qa_chain.invoke({"query": user_question})
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
    except Exception as e:
        st.error(f"Error processing query: {e}")
        print(f"Query error: {e}")  # Debug

# Clean up temporary file
if os.path.exists("temp.pdf"):
    os.remove("temp.pdf")