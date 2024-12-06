import streamlit as st
from dotenv import load_dotenv
from utils import user_input, get_pdf_text, get_text_chunks, get_vector_store
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Set up the API key
genai.configure(api_key="AIzaSyAV3yNNh9u9nnBf-XQRD642OL9rDmlKobQ")

# Test the connection
try:
    models = genai.list_models()
    print("Available Models:", models)
except Exception as e:
    print("Error:", e)


# Configure Streamlit page
st.set_page_config(page_title="Document Genie", layout="wide")

def main():
    st.header("📄 Document Genie: Your AI Clone Chatbot")
    
    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader(
            "Upload your PDF files:", 
            accept_multiple_files=True, 
            type=["pdf"],
            key="pdf_uploader"
        )
        
        if st.button("Submit & Process", key="process_button"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        # Extract text, split into chunks, and create vector store
                        raw_text = get_pdf_text(pdf_docs)
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing completed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
            else:
                st.warning("Please upload PDF files before clicking submit.")

    # Main section for user interaction
    st.subheader("Ask a Question")
    user_question = st.text_input("Type your question:", key="user_question")
    if user_question:
        with st.spinner("Generating answer..."):
            try:
                response = user_input(user_question)
                st.write("### Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while answering your question: {e}")

    

if __name__ == "__main__":
    main()
