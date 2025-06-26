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
from langchain_community.document_loaders import UnstructuredURLLoader
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
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": "cpu"}
)
vector_db = FAISS.load_local("faiss_index1", embeddings,allow_dangerous_deserialization=True)

retriever = vector_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 4}
)
os.environ["GROQ_API_KEY"] = "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"
os.environ["TAVILY_API_KEY"] = "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"
llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct",temperature=0.2, max_tokens=700, streaming=True)


# ───── Semantic NIT Website Search ──────────
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")

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

# ───── Memory Setup ─────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ───── Tools Setup ──────────────────────────
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, memory=memory)

tools = [
    Tool(name="RAG_KnowledgeBase", func=rag_chain.run,
         description="Use this to answer factual queries from NIT PDFs."),
    Tool(name="Website_Semantic_Search", func=search_college_website_semantic,
         description="Use this to find forms, links, or notices from the NIT website."),
    Tool(name="Tavily_Web_Search", func=TavilySearchResults().run,
     description="Use only for NIT Trichy topics not found in KB. Do not search for other topic")

]
from langchain.schema import SystemMessage



# ───── User Query Rewriter Chain ─────
rewrite_prompt = PromptTemplate(
    input_variables=["query"],
    template=(
        "You are a helpful assistant. Rewrite the user's question to make it clearer and more specific, "
        "but keep the meaning the same. Query: {query}"
    )
)
rewrite_chain = LLMChain(llm=llm, prompt=rewrite_prompt)


system_prompt = (
    "You are an intelligent and helpful AI assistant. Your purpose is to provide accurate, reliable, and concise answers "
    "specifically about the National Institute of Technology, Trichy (NIT Trichy).\n\n"
    "📌 Instructions:\n"
    "- If the user's question is **not related to NIT Trichy**, politely respond with:\n"
    "  'I'm designed to answer only NIT Trichy-related queries.'\n"
    "- If you are **uncertain** whether a question is related to NIT Trichy, say:\n"
    "  'Sorry, I can only help with questions specifically related to NIT Trichy.'\n"
    "- Do not guess or provide general world knowledge not tied to NIT Trichy.\n"
    "- Be polite, concise, and helpful at all times.\n\n"
    "Now, here is the user’s query:"
)


# ───── Agent Executor with Tools + Memory + System Prompt ─────
agent_executor = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    agent_kwargs={"system_message": SystemMessage(content=system_prompt)},
    memory=memory
)

# ───── Chatbot Function with Rewriting ─────

    
def chatbot_response(user_query):
    try:
        rewritten_query = rewrite_chain.run({"query": user_query})
        full_prompt = f"{system_prompt}\n{rewritten_query}"
        response = agent_executor.run(input=full_prompt)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}" 


import whisper
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
import streamlit as st
from streamlit_mic_recorder import mic_recorder

st.set_page_config(page_title="NIT Trichy Chatbot", page_icon="🎓", layout="wide")
st.title("🎓 NIT Trichy Chatbot")
st.write("Ask me anything about NIT Trichy, including admissions, academics, and campus life!")

# ─── Initialize Chat History ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── Display Chat History ────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ─── Capture Input: Voice or Text ────────────────────────
text_input = st.chat_input("Or type your question here...")
audio_bytes = mic_recorder(
    start_prompt="🎤 Speak",
    stop_prompt="🛑 Stop",
    just_once=True,
    key=f"mic_{len(st.session_state.messages)}"
)

user_input = None
if audio_bytes:
    with st.spinner("Transcribing your voice..."):
        user_input = transcribe_audio_local(audio_bytes)
        st.success(f"📝 Transcribed: {user_input}")
elif text_input:
    user_input = text_input

# ─── Run Chatbot Logic and Display Response ──────────────
if user_input:
    # Append user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run chatbot response
    with st.spinner("Loading..."):
        response = chatbot_response(user_input)

    # Append assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
        speak_text(response)
