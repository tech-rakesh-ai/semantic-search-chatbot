# Semantic Search and Conversational ChatBot with Google Gemini & Elasticsearch

This project is a web-based application that allows you to upload employee data in JSON format, store it in Elasticsearch, and perform semantic search and conversation using Google Gemini’s Generative AI (LLM) through LangChain. The application is built using **Streamlit** for the user interface.

## Key Features

1. **Employee Data Storage in Elasticsearch**: 
   - Upload employee data in JSON format.
   - Store the data in an Elasticsearch index for future retrieval.
   - Each record is embedded using Google's Gemini AI model for efficient similarity-based search.

2. **Semantic Search & Conversational AI**: 
   - Ask questions about the employee data.
   - Use Google Gemini's Generative AI (ChatGoogleGenerativeAI) for question-answering.
   - The system retrieves relevant records from Elasticsearch using similarity search and provides answers based on the retrieved data.

3. **Real-time Conversational Interface**:
   - Chat interface for user interaction.
   - Maintains conversation history and offers continuous interaction with context-aware responses.
   - Embeds user questions and provides responses using the **ConversationalRetrievalChain** from LangChain.

## How It Works

### 1. **Upload Employee Data**:
   - The application allows you to upload a JSON file containing employee data.
   - It verifies that the uploaded file contains a list of records and processes each record for embedding and storage in Elasticsearch.

### 2. **Embedding Data into Elasticsearch**:
   - After uploading the data, users can click the "Embed Data" button to process and store the employee records in Elasticsearch.
   - The application uses Google Gemini's Embedding model (`embedding-001`) to convert the records into embeddings, enabling efficient similarity-based retrieval.
   - Each record is added to Elasticsearch with its embedding, and the progress is displayed with a progress bar.

### 3. **Conversational AI**:
   - Users can ask questions related to the stored employee data through the chat interface.
   - The application uses **ConversationalRetrievalChain** to retrieve relevant information from Elasticsearch and provide answers.
   - It maintains chat history and continues to build context through conversation.

### 4. **Session Memory**:
   - The application maintains chat history using **ConversationBufferMemory** to ensure continuity in conversation.
   - This allows the chatbot to reference previous questions and answers during the conversation.

## Technologies Used

- **Streamlit**: For building the user interface and handling user interactions.
- **LangChain**: For connecting the LLM (Google Gemini AI) and Elasticsearch for embedding and retrieval.
  - `GoogleGenerativeAIEmbeddings`: Embedding the employee data.
  - `ChatGoogleGenerativeAI`: Using Google Gemini's `gemini-1.5-pro` model for conversational AI.
  - `ConversationalRetrievalChain`: Handles conversation flow and retrieval from Elasticsearch.
- **Elasticsearch**: Stores the employee data with embeddings for efficient similarity-based search.
- **Google Gemini Pro API**: Provides the LLM (Language Model) capabilities for embedding and conversational AI.
- **dotenv**: For loading environment variables (API keys, Elasticsearch credentials).

## How to Use

### 1. **Clone the Repository**
```bash
git clone https://github.com/tech-rakesh-ai/semantic-search-chatbot.git
cd semantic-search-chatbot
```

### 2. **Install the Dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Set up Environment Variables**
Create a `.env` file in the project root directory and add your API keys and Elasticsearch credentials:
```
GOOGLE_API_KEY=your_google_gemini_api_key
ES_URL=https://your_elasticsearch_url
ES_USER=your_elasticsearch_username
ES_PASSWORD=your_elasticsearch_password
INDEX_NAME=your_index_name
```

### 4. **Run the Application**
```bash
streamlit run app.py
```

### 5. **Upload Employee Data**
- Once the app is running, use the sidebar to upload a JSON file containing employee data.
- The data should be in the format of a list of JSON objects (records).

### 6. **Start Chatting**
- After embedding the data, use the chat interface to ask questions about the uploaded employee data.
- The chatbot will provide intelligent, context-aware responses based on the data.

## Sample Employee Data JSON

The uploaded JSON file should contain a list of employee records in the following format:
```json
[
  {
    "name": "John Doe",
    "position": "Software Engineer",
    "age": 30,
    "department": "IT"
  },
  {
    "name": "Jane Smith",
    "position": "Data Scientist",
    "age": 28,
    "department": "Data Science"
  }
]
```

## Troubleshooting

- Ensure your Google API key and Elasticsearch credentials are correctly configured in the `.env` file.
- Make sure the uploaded JSON file contains a valid list of records.

## Logging

The application uses Python's `logging` library to log information for debugging purposes. The logging level is set to `DEBUG`. You can change the logging level in the `app.py` file if needed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
