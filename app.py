import streamlit as st
import requests
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import numpy as np
import faiss
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
import google.generativeai as genai
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GENAI_API_KEY"))

MODEL_DIR = Path("./saved_models")
MODEL_DIR.mkdir(exist_ok=True)

question_encoder_path = MODEL_DIR / "dpr_question_encoder"
context_encoder_path = MODEL_DIR / "dpr_context_encoder"
question_tokenizer_path = MODEL_DIR / "dpr_question_tokenizer"
context_tokenizer_path = MODEL_DIR / "dpr_context_tokenizer"

if question_encoder_path.exists():
    question_encoder = DPRQuestionEncoder.from_pretrained(str(question_encoder_path))
else:
    question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_encoder.save_pretrained(str(question_encoder_path))

if context_encoder_path.exists():
    context_encoder = DPRContextEncoder.from_pretrained(str(context_encoder_path))
else:
    context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_encoder.save_pretrained(str(context_encoder_path))

if question_tokenizer_path.exists():
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(str(question_tokenizer_path))
else:
    question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
    question_tokenizer.save_pretrained(str(question_tokenizer_path))

if context_tokenizer_path.exists():
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(str(context_tokenizer_path))
else:
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
    context_tokenizer.save_pretrained(str(context_tokenizer_path))

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

def encode_context(content, max_length=512):
    inputs = context_tokenizer(content, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    context_embedding = context_encoder(**inputs).pooler_output.detach().numpy()
    return context_embedding

def encode_query(query, max_length=512):
    inputs = question_tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
    return query_embedding

def search(query_embedding, context_embeddings, k=5):
    similarities = np.dot(context_embeddings, query_embedding.T)
    closest_idx = similarities.argsort(axis=0)[-k:][::-1]
    return closest_idx

def get_relevant_text(query, context, max_results=2):
    context_embeddings = [encode_context(c) for c in context]
    query_embedding = encode_query(query)

    closest_idx = search(query_embedding, np.array(context_embeddings).squeeze(), k=max_results)
    closest_idx = closest_idx.flatten()
    relevant_texts = [context[i] for i in closest_idx]
    return relevant_texts

def bot_response(model, query, relevant_texts):
    context = ' '.join(relevant_texts)
    prompt = f"This is the context of the document\nContext: {context}\nAnd this is the user query\nUser: {query}\nAnswer the query with respect to the context provided like a human having a lot of knowledge on the context provided\nBot:"
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.7,
            max_output_tokens=150
        )
    )
    return response.text

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

        st.session_state.messages.append({
            'role': 'assistant',
            'content': "All the information is retrieved, ask your queries, and start the chat!"
        })
    except Exception as e:
        st.error(f"An error occurred: {e}")

if st.session_state.website_paragraphs:
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-002")

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
                relevant_texts = get_relevant_text(user_question, st.session_state.website_paragraphs)
                bot_reply = bot_response(model, user_question, relevant_texts)

            row_a = st.columns(2)
            row_a[0].chat_message('assistant').markdown(bot_reply)

            st.session_state.messages.append(
                {'role': 'assistant',
                 'content': bot_reply}
            )

    except Exception as e:
        st.chat_message('assistant').markdown('There might be an error, try again')
        st.session_state.messages.append(
            {
                'role': 'assistant',
                'content': 'There might be an error, try again'
            }
        )
