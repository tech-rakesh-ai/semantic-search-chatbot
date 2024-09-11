import streamlit as st
import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import ElasticsearchStore
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_elasticsearch import ElasticsearchStore
from langchain.schema import SystemMessage, HumanMessage

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
load_dotenv()

# Access the Google API key
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Initialize Google Generative AI Embedding
embeddings_llm = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)

# Initialize Elasticsearch connection
es = Elasticsearch(hosts=["http://localhost:9200"], basic_auth=("elastic", "Xu*0DmHWEHvRoTWFA=Vs"))


# Load your JSON data
@st.cache_data
def load_data():
    return [
        {"id": 121, "Name": "Ramesh", "Address": "123 Main St, Mumbai",
         "About": "Software Engineer with 5 years of experience in full-stack development."},
        {"id": 122, "Name": "Rakesh", "Address": "456 Elm St, Delhi",
         "About": "Marketing specialist focused on digital campaigns and brand management."},
        {"id": 123, "Name": "Rahul", "Address": "789 Maple St, Bangalore",
         "About": "Finance manager with a background in corporate finance and investment banking."},
        {"id": 124, "Name": "Ramu", "Address": "101 Oak St, Hyderabad",
         "About": "Operations manager with expertise in supply chain management and logistics."},
        {"id": 125, "Name": "Raju", "Address": "202 Pine St, Chennai",
         "About": "Project manager with a focus on IT infrastructure and cloud computing."},
        {"id": 126, "Name": "Rohit", "Address": "303 Cedar St, Pune",
         "About": "Data analyst with a passion for big data and machine learning."},
        {"id": 127, "Name": "Ravi", "Address": "404 Birch St, Kolkata",
         "About": "Graphic designer specializing in UI/UX design and visual branding."},
        {"id": 128, "Name": "Rina", "Address": "505 Willow St, Ahmedabad",
         "About": "Content writer and editor with expertise in SEO and digital marketing."},
        {"id": 129, "Name": "Rajesh", "Address": "606 Palm St, Jaipur",
         "About": "Sales executive with a strong background in B2B sales and customer relations."},
        {"id": 130, "Name": "Rita", "Address": "707 Spruce St, Lucknow",
         "About": "Human resources professional with experience in talent acquisition and employee engagement."}
    ]


# # Check if index exists, and create it with correct mappings if it doesn't
# def create_index():
#     if not es.indices.exists(index="emp_data"):
#         es.indices.create(
#             index="emp_data",
#             body={
#                 "settings": {
#                     "number_of_shards": 1,
#                     "number_of_replicas": 1
#                 }
#             }
#         )
#         st.success("Index 'emp_data' created successfully.")
#     else:
#         st.info("Index 'emp_data' already exists.")
#
#
# def embed_and_index_data(data):
#     for record in data:
#         text_to_embed = ' '.join([
#             record.get("Name"),
#             record.get("Address"),
#             record.get("About"),
#         ])
#         embedding_vector = embeddings_llm.embed_query(text_to_embed)
#         record["vector"] = embedding_vector
#         try:
#             es.index(index="emp_data", body=record)
#             print(f"Record {record} Stored successfully!")
#         except Exception as e:
#             st.error(f"Error indexing document: {str(e)}")
#             return False
#     st.success("Data indexed successfully with embeddings.")
#     return True


# Custom ElasticsearchStore with modified document builder
# class CustomElasticsearchStore(ElasticsearchStore):
#     @staticmethod
#     def custom_doc_builder(hit: dict) -> Document:
#         return Document(
#             page_content=hit["_source"]["content"],
#             metadata=hit["_source"]["metadata"]
#         )
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.client.options(ignore_status=[400, 404])

# Initialize the chatbot

def init_chatbot():
    # Initialize the vectorstore using Elasticsearch
    vectorstore = ElasticsearchStore(
        es_url="http://localhost:9200",  # Elasticsearch URL
        index_name="emp_data",  # Index name in Elasticsearch
        embedding=embeddings_llm,  # Predefined embedding model (e.g., OpenAI embeddings)
        es_user="elastic",  # Elasticsearch username
        es_password="Xu*0DmHWEHvRoTWFA=Vs"  # Elasticsearch password
    )
    # Create a retriever from the vectorstore
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize the LLM with Google Gemini Pro API
    google_api_key = os.getenv("GOOGLE_API_KEY")  # Load API key from environment variable
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=google_api_key)

    # Initialize memory to store chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    # Use the built-in ConversationalRetrievalChain (CRIE)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        # memory=memory,  # Memory keeps track of chat history
        return_source_documents=True,  # Optionally return source documents
        return_generated_question=True
    )

    return qa_chain, retriever, memory


def check_index_content():
    # # Create the full retrieval chain
    # qa_chain = create_retrieval_chain(
    #     retriever=history_aware_retriever,
    #     question_answer_chain=question_answer_chain,
    #     input_variables=["input", "context"]  # Ensure it accepts context
    # )
    result = es.count(index="emp_data")
    count = result['count']
    st.write(f"Number of documents in index: {count}")
    if count == 0:
        st.warning("The index is empty. Please index your data first.")


# Streamlit UI
st.title("Conversation Analytics With Elasticsearch")

# Sidebar for data operations
with st.sidebar:
    st.header("Data Operations")
    if st.button("Create Index"):
        create_index()

    if st.button("Embed and Index Data"):
        data = load_data()
        if embed_and_index_data(data):
            st.success("Data embedded and indexed successfully!")
        else:
            st.error("Failed to embed and index data. Check the logs for more information.")

    if st.button("Check Index Content"):
        check_index_content()

# Main chat interface
st.header("Chat with the Employee Data")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the employees"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    chat_history = []
    chatbot, retriever, memory = init_chatbot()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Prepare the input for the chatbot
            response = chatbot.invoke({"question": prompt, "chat_history": chat_history})
            # chat_history.append((prompt, res["answer"]))
            answer = response['answer']
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    # Update memory with the new exchange
    # memory.chat_memory.add_user_message(prompt)
    # memory.chat_memory.add_ai_message(response)

# Add a button to clear chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.success("Chat history cleared!")


