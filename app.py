import streamlit as st
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import numpy as np
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 200)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks,embeddings):
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_relevant_text(user_query, embeddings):
    new_db = FAISS.load_local("faiss_index",embeddings)
    docs = new_db.similarity_search(user_query)
    context = docs[0].page_content

    return context

def fetch_website_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        content = ' '.join([p.text for p in soup.find_all('p')])
        return content, soup
    else:
        raise Exception(f"Failed to fetch website: {response.status_code}")

def crawl_website(base_url, max_pages=10):
    visited = set()
    to_visit = [base_url]
    all_content = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            content, soup = fetch_website_content(current_url)
            all_content.append(content)
            visited.add(current_url)

            for link in soup.find_all('a', href=True):
                href = urljoin(base_url, link['href'])
                if urlparse(href).netloc == urlparse(base_url).netloc:
                    to_visit.append(href)
        except Exception as e:
            st.error(f"Error fetching {current_url}: {e}")

    return " ".join(all_content)


def bot_response(model, query, relevant_texts):
    prompt = f"""This is the context of the document 
    Context: {relevant_texts}
    And this is the user query
    User: {query}
    Answer the query with respect to the context provided (you can use upto 35% of additional knowledge too),
    like a professional having a lot of knowledge on the context provided
    Bot:
    """
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            # max_output_tokens=150
        )
    )
    return response.text

st.set_page_config(page_title="Website ChatBot", layout="wide")
st.title("Website Chatbot")
st.markdown("Enter the URL of a website to start chatting with the content!")

if "website_paragraphs" not in st.session_state:
    st.session_state.website_paragraphs = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "previous_url" not in st.session_state:
    st.session_state.previous_url = ""

with st.sidebar:
    st.header("Enter the URL for the Website")
    base_url = st.text_input("Enter Website URL:")

    if base_url != st.session_state.previous_url:
        st.session_state.messages = []  
        st.session_state.website_paragraphs = None  

    if st.button("Submit"):
        st.session_state.previous_url = base_url  

if base_url and st.session_state.website_paragraphs is None:
    try:
        with st.spinner("Crawling the website. Please wait..."):
            website_content = crawl_website(base_url)

        st.session_state.website_paragraphs = website_content.split('\n')
        
        if website_content:
            chunks = get_text_chunks(website_content)
            get_vector_store(chunks, embeddings)
            st.success("Crawled the website successfully!")

        st.session_state.messages.append({
            'role': 'assistant',
            'content': "All the information is retrieved, ask your queries, and start the chat!"
        })
        
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.session_state.website_paragraphs:
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")

    for message in st.session_state.messages:
        row = st.columns(2)
        if message['role'] == 'user':
            row[1].chat_message(message['role']).markdown(message['content'])
        else:
            row[0].chat_message(message['role']).markdown(message['content'])

    try:
        user_question = st.chat_input("Enter your query here !!")

        if user_question:
            row_u = st.columns(2)
            row_u[1].chat_message('user').markdown(user_question)
            st.session_state.messages.append(
                {'role': 'user',
                 'content': user_question}
            )

            with st.spinner("Generating response..."):
                relevant_texts = get_relevant_text(user_question, embeddings)
                bot_reply = bot_response(model, user_question, relevant_texts)

            row_a = st.columns(2)
            row_a[0].chat_message('assistant').markdown(bot_reply)

            st.session_state.messages.append(
                {'role': 'assistant',
                 'content': bot_reply}
            )

    except Exception as e:
        st.chat_message('assistant').markdown(f'There might be an error, try again {str(e)}')
        st.session_state.messages.append(
            {
                'role': 'assistant',
                'content': f'There might be an error, try again {str(e)}'
            }
        )
