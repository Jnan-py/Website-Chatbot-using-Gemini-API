# Website Chatbot AI

Website Chatbot AI is a Streamlit-based application that allows you to chat with the content of any website. By entering a website URL, the app crawls and extracts the textual content from the site, splits the content into manageable chunks, indexes it using a FAISS vector store, and then uses Google Generative AI to answer your queries in a conversational manner.

## Features

- **Website Crawling & Text Extraction:**  
  The app crawls the specified website and extracts text content from all paragraph elements using BeautifulSoup.

- **Content Chunking & Indexing:**  
  Extracted text is split into chunks using LangChain's RecursiveCharacterTextSplitter, and then indexed with FAISS for efficient similarity search.

- **Generative AI Chatbot:**  
  Uses Google Generative AI (Gemini models) to generate context-aware responses based on the website content and user queries.

- **Interactive Chat Interface:**  
  Engage in a conversation with the AI where previous messages are maintained to provide context.

- **Easy Setup:**  
  Simply provide a website URL in the sidebar, and the app handles the rest!

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jnan-py/Website-Chatbot-using-Gemini-API.git
   cd Website-Chatbot-using-Gemini-API
   ```

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   - Create a `.env` file in the project root with the following keys:
     ```
     GOOGLE_API_KEY=your_google_gemini_api_key_here
     ```

## Usage

1. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

2. **Enter a Website URL:**

   - In the sidebar, input the URL of the website you want to chat about and click **Submit**.
   - The app will crawl the site, extract the text, split it into chunks, and build a FAISS index for similarity search.

3. **Start Chatting:**
   - Once the content is processed, use the chat input to ask questions about the website.
   - The AI will retrieve relevant context and generate responses based on both the website content and additional knowledge.

## Project Structure

```
website-chatbot-ai/
│
├── app.py                 # Main Streamlit application file
├── downloads/             # Directory where downloaded files are saved
├── .env                   # Environment variables file (create and add your API keys)
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

## Technologies Used

- **Streamlit:** For building the interactive web application.
- **BeautifulSoup & Requests:** For web crawling and HTML content extraction.
- **LangChain:**
  - _Text Splitters_ for breaking content into chunks.
  - _Community Vector Store (FAISS)_ for indexing text and performing similarity searches.
- **Google Generative AI (Gemini):** For generating context-aware responses.
- **Python-Dotenv:** For managing environment variables.
- **NumPy:** For numerical operations.

Save these files in your project directory. To launch the app, run:

```bash
streamlit run app.py
```

Feel free to adjust the documentation as needed. Enjoy chatting with your Website Chatbot AI!

