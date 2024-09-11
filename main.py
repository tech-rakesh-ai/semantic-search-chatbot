import streamlit as st
import json
import os
from langchain_core.messages import AIMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_elasticsearch import ElasticsearchStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()

# Set up API keys
google_api_key = os.getenv("GOOGLE_API_KEY")
es_url = os.getenv("ES_URL")
es_user = os.getenv("ES_USER")
es_password = os.getenv("ES_PASSWORD")
index_name = "emp_data"

# Initialize the embedding LLM
embeddings_llm = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                              google_api_key=google_api_key)

# Initialize the LLM with Google Gemini Pro API
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

# Initialize ElasticsearchStore
es_store = ElasticsearchStore(
    es_url=es_url,
    index_name=index_name,
    embedding=embeddings_llm,
    es_user=es_user,
    es_password=es_password
)


# Function to store employee data into Elasticsearch
def store_in_elasticsearch(employee_data):
    text = json.dumps(employee_data)
    es_store.add_texts(
        texts=[text],
        metadatas=[employee_data],
        refresh_indices=True
    )


# Conversation Section
st.sidebar.title("Semantic Search and Conversation: ChatBot")

# Upload Section for Employee Data JSON file
st.sidebar.subheader("Upload Employee Data JSON File")
uploaded_file = st.sidebar.file_uploader("Choose a JSON file", type="json")

if uploaded_file:
    try:
        # Load the file content as JSON
        employee_data = json.load(uploaded_file)

        # Check if the data is a list of records
        if isinstance(employee_data, list):
            # Embed Data button
            if st.sidebar.button("Embed Data"):
                with st.spinner("Embedding data... Please wait."):
                    progress_bar = st.sidebar.progress(0)
                    total_records = len(employee_data)

                    for idx, record in enumerate(employee_data):
                        store_in_elasticsearch(record)
                        # Update progress bar
                        progress_bar.progress((idx + 1) / total_records)
                st.sidebar.success(f"{total_records} records successfully stored in Elasticsearch.")
        else:
            st.sidebar.error("The uploaded file does not contain a list of records.")
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON format. Please upload a valid JSON file.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def handle_conversation(user_input):
    # Initialize the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=es_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        memory=st.session_state.chat_history
    )
    result = qa_chain.invoke({"question": user_input})
    logging.debug(f"User Input: {user_input}")
    logging.debug(f"Result from QA Chain: {result}")

    # Create an AIMessage object with the response
    response = AIMessage(content=result["answer"])

    return response


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about employee data:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = handle_conversation(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response.content)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response.content})
