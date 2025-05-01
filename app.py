import os
import json
import pickle
import requests
import bs4
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, LLMChain, StuffDocumentsChain
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import HTMLHeaderTextSplitter
from sentence_transformers import SentenceTransformer, util
from urllib.parse import urljoin
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer, util
from bs4 import BeautifulSoup
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from gtts import gTTS
import tempfile
import pygame
import uuid
import whisper

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_db = FAISS.load_local("faiss_index1", embeddings,allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct",temperature=0.2, max_tokens=700, streaming=True)


# ───── Semantic NIT Website Search ──────────
embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def search_college_website_semantic(query):
    base_url = "https://www.nitt.edu"
    visited_pages = set()
    link_texts = []
    links = []

    def scrape_page(url):
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.text, "html.parser")
        except Exception:
            return
        for link in soup.find_all("a"):
            text = link.text.strip()
            href = link.get("href")
            if text and href:
                full_link = urljoin(base_url, href)
                if full_link not in visited_pages and base_url in full_link:
                    visited_pages.add(full_link)
                    link_texts.append(text)
                    links.append(full_link)

    scrape_page(base_url)
    internal_pages = [
        "https://www.nitt.edu/home/academics/curriculum/",
        "https://www.nitt.edu/home/academics/fees_section/",
    ]
    for page in internal_pages:
        scrape_page(page)

    if not link_texts:
        return "No relevant links found on the site."

    link_embeddings = embedder.encode(link_texts, convert_to_tensor=True)
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, link_embeddings)[0]
    top_results = cos_scores.topk(k=3)

    results = []
    for score, idx in zip(top_results.values, top_results.indices):
        results.append(f"{link_texts[idx]} ➤ {links[idx]}")
    return "\n".join(results)

# ───── Tools Setup ──────────────────────────
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

tools = [
    Tool(name="RAG_KnowledgeBase", func=rag_chain.run,
         description="Use this to answer factual queries from NIT PDFs."),
    Tool(name="Website_Semantic_Search", func=search_college_website_semantic,
         description="Use this to find forms, links, or notices from the NIT website."),
    Tool(name="Tavily_Web_Search", func=TavilySearchResults().run,
         description="Use this to search live internet if RAG fails.")
]
from langchain.schema import SystemMessage
system_prompt = (
    "You are an intelligent AI assistant specifically trained to answer queries related to the National Institute of Technology, Trichy (NIT Trichy). "
    "Use only reliable and relevant sources such as the official website, academic documents, and FAQ knowledge base to answer. "
    "If a question is unrelated to NIT Trichy, politely decline to answer or redirect the user appropriately. "
    "Always answer in a clear, friendly, and helpful tone."
)
agent_executor = initialize_agent(
    tools=tools,  # Define your tools
    llm=llm,      # Define your language model
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": system_prompt}
)


# ─── CHATBOT FUNCTION ─────────────────────────────────────
def chatbot_response(user_query):
    try:
        return agent_executor.run(user_query)
    except Exception as e:
        return f"❌ Error: {str(e)}"


model = whisper.load_model("base")  # or "tiny", "small", etc.

def transcribe_audio_local(audio_dict):
    audio_data = audio_dict.get("bytes") if isinstance(audio_dict, dict) else audio_dict
    if not audio_data:
        return "No audio data received."

    with open("temp.wav", "wb") as f:
        f.write(audio_data)

    result = model.transcribe("temp.wav")
    return result["text"]


def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        st.audio(fp.name, format="audio/mp3")


# ─── STREAMLIT UI ─────────────────────────────────────────
st.set_page_config(page_title="NIT Trichy Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 NIT Trichy Chatbot")
st.write("Ask me anything about NIT Trichy, including admissions, academics, and campus life!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
from streamlit_mic_recorder import mic_recorder


audio_bytes = mic_recorder(
    start_prompt="🎤 Speak",
    stop_prompt="🛑 Stop",
    just_once=True,
    key=f"mic_{len(st.session_state.messages)}"
)

# Fallback text input
text_input = st.chat_input("Or type your question here...")

# Decide user input source
user_input = None
if audio_bytes:
    with st.spinner("Transcribing your voice..."):
        user_input = transcribe_audio_local(audio_bytes)
        st.success(f"📝 Transcribed: {user_input}")
elif text_input:
    user_input = text_input

# Proceed with chat if any input exists
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("Thinking..."):
        response = chatbot_response(user_input)

    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        speak_text(response) 

