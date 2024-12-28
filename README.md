# Website Chatbot

This project integrates AI in order to create a chatbot which can answer queries from the user by making use of the content fetched from a given website. The process of fetching content is done through a method called **web crawling**, which fetches the data from the entered URL and its subpages. Then, the chatbot processes the fetched information and produces relevant responses based on user input.

## Project Overview

The core functionality of this chatbot includes:

- **Website Crawling**: Upon entering the URL and clicking the "Submit" button, the program crawls the website and fetches all textual content from pages and their subpages.
- **AI-Driven Response**: The fetched website content then gets indexed and the information is used by the AI chatbot to give answers to the queries entered into the website accurately.
- **Interaction**: The user can ask anything that he/she wants, and the chatbot has to generate a human-like response regarding the content found on the website.

### Key Features

- Web Crawling: Fetching the text content from that website and all its related subpages.
- Question-Answering System: Using AI models to perceive the user's query in order to return the best text related to the current query from the website.
- Chat Interface: Here, the user interacts through a simple chat interface with the bot.

## Installation

To run this project, follow the steps below:

### Clone the Repository

You can clone this repository to your local machine by using the following command.

```bash
git clone <repository_url>
```

### Install Dependencies

You will need Python 3.7+ and the following Python libraries. You can install them with pip.

```bash
pip install -r requirements.txt
```

### Set up Environment Variables

This project utilizes the Google Generative AI API. You will need to add the API key in the `.env` file. Create a `.env` file in the root and add your API key here:

```
GENAI_API_KEY=your_google_api_key_here
```

### Dependencies

These are the dependencies used within this project:

- `beautifulsoup4`: HTML content is parsed and text is pulled from web pages.
- `numpy`: Numerical computations and operations.
- `faiss-cpu`: For efficient similarity search of context embeddings.
- `transformers`: The Hugging Face library, used to load pre-trained models for question and context encoding.
- `requests`: For sending HTTP requests to fetch content from websites.
- `sentence-transformers`: Handling sentence embeddings, used to encode website content and user queries.
- `google-generativeai`: Used to integrate the Google Generative AI model to generate responses from the chatbot.
- `python-dotenv`: To load environment variables from a `.env` file.

## Usage

1. **Start the Streamlit App:**

Run the following command to start the web application:

```bash
streamlit run app.py
```

2. **Enter Website URL**:

- Once the app has launched, open it up in your browser.
- Within the sidebar, fill out a URL for a website you'd like your chatbot to learn from.

3. **Engage with the Chatbot**:

- After pressing "Submit", the bot will crawl the website and collect the content.
- You can now begin querying, and the chatbot will generate responses based on the content it has crawled.

## Methodology & Workflow

### Crawling the Website

- Crawling the website is a process that gets initiated with the entry of a URL and submission.
  Fetches all the HTML content from the pages by using `requests` module and parsing it with `BeautifulSoup` to extract the text.
- **Fetching Subpages**: The crawler further fetches all the subpages on the website since that is the only other kind of internal link remaining to be fetched. To achieve this, it simply looks for `href` attributes in `<a>` tags and recursively visits them.

- **Content Encoding**: The content of the website (textual data) is encoded using pre-trained models. `DPRContextEncoder` encodes the context, while `DPRQuestionEncoder` encodes the queries entered by the user.

This entails a **Similarity Search**: Using cosine similarity in the embedding of the user's query and context content, a chatbot searches for similar text segments related to its query.

Relevant text is then processed by the **Generative AI Model** from Google via the `google-generativeai` API; it leverages the contextual information for generating a sensible response towards the user's query.

1. User inputs: "What is the company's mission?"
2. Bot Crawls the website: Retrieves the company's About Us page and all relevant sections.
3. User asks: "What is the mission of the company?"
4. Bot Response: "The company's mission is to innovate in technology and customer service.

- Better handling of dynamic websites. Crawling is done only on static pages now. It can be done way better with headless browsers in case of very dynamic sites with heavy usage of JavaScript.
- Scaling the crawling logic for very big sites so that not every part of the website gets crawled.
- Integrate more advanced AI models so that it has better contextual understanding and conversation flow.
