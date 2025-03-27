import os
import json
import bs4
import pickle
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from langchain.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
faq_vector_db = FAISS.load_local("faq_index", embeddings,allow_dangerous_deserialization=True)


# Create Retrievers
retriever = vector_db.as_retriever(search_type="mmr")
faq_retriever = faq_vector_db.as_retriever(search_type="mmr")

os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3, max_tokens=900, streaming=True)

# Define Prompt for Structured Responses
prompt_template = PromptTemplate.from_template("""
You are a helpful AI assistant for NIT Trichy. Your job is to provide **accurate, concise, and structured** answers based on FAQs and retrieved documents.
                                               
### FAQs Context:
{faq_context}

### Retrieved Documents:
{document_context}

### User Question:
{question}

### Answer:
""")


llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def retrieve_faq(user_question, k=3):
    """Retrieve the top-k most relevant FAQs based on similarity search."""
    similar_faqs = faq_retriever.get_relevant_documents(user_question, search_kwargs={"k": k})
    
    if similar_faqs:
        return "\n\n".join([faq.page_content for faq in similar_faqs])  # Combine FAQs
    return "No relevant FAQ found."  # Default response

def chatbot_response(user_question):
    """Integrate FAQs + document retrieval + LLM for structured responses."""
    try:
        retrieved_faq = retrieve_faq(user_question)
        retrieved_docs = retriever.get_relevant_documents(user_question)
        
        structured_prompt = prompt_template.format(
            faq_context=retrieved_faq,
            document_context="\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant document found.",
            question=user_question
        )
        
        response = llm_chain.invoke({
            "faq_context": retrieved_faq,
            "document_context": "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant document found.",
            "question": user_question
        })
        
        # Convert response to plain text if necessary
        return response if isinstance(response, str) else response.get("text", "I couldn't generate a response.")
    except Exception as e:
        return f"Error generating response: {str(e)}"


st.set_page_config(page_title="NIT Trichy Chatbot", page_icon="🎓", layout="wide")

st.title("🎓 NIT Trichy Chatbot")
st.write("Ask me anything about NIT Trichy, including admissions, academics, and campus life!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Ask your question here...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    response = chatbot_response(user_input)

    # Add chatbot response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chatbot response
    with st.chat_message("assistant"):
        st.markdown(response)