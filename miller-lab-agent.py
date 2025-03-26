import os
import tempfile
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate

import csv
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate('miller-lab-agent-firebase-adminsdk-r1vbq-3fdf45cbea.json')
    firebase_admin.initialize_app(cred)

# Get Firestore database instance
db = firestore.client()

openai_api_key = st.secrets["shreyas_openai_api_key"]

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

@st.cache_resource(ttl="1h")
def load_documents_from_csv(csv_filename):
    documents = []
    with open(csv_filename, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            page_content = row['Page Content']
            metadata = {
                'labeled_page_number': row['Labeled Page Number'],
                'page_header': row['Page Header'],
                'page_link': row['Page Link']
            }
            document = Document(page_content, metadata)
            documents.append(document)
    return documents

@st.cache_resource(ttl="1h")
def loading_handbook(csv_filename):
    document_splits = load_documents_from_csv(csv_filename)
    openai_key = openai_api_key

    # VectorDB setup
    embedding = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = Chroma.from_documents(documents=document_splits, embedding=embedding)

    # Define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    return retriever

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None
        self.initial_prompt = ""

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        if prompts and len(prompts) > 0:
            self.initial_prompt = prompts[0]
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Retrieving Handbook Information**")
        self.content = "**Handbook Links:**\n\n"
        self.headings = []

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            page_number = doc.metadata.get('labeled_page_number', 'Unknown Page Number')
            page_header = doc.metadata.get('page_header', 'Unknown Header')
            page_link = doc.metadata['page_link']
            source = f"[{page_header}, Page {page_number}]({page_link})"  
            self.headings.append(f"{page_header}, Page {page_number}")
            self.content += f"**Document {idx + 1} from {source}:**\n{doc.page_content}\n\n"
            self.status.markdown(source)

        self.status.update(state="complete")
        return self.headings

def save_conversation_to_firestore(user_input, assistant_response, pages_returned):
    conversation_ref = db.collection("conversations").document()
    conversation_ref.set({
        'user_input': user_input,
        'assistant_response': assistant_response,
        'pages_returned': pages_returned,
        'timestamp': firestore.SERVER_TIMESTAMP
    })

st.title('👪 Caregiving Handbook')
st.markdown('🙏🏼 Welcome to the Handbook Healthcare Assistant!')

st.markdown("We have provided the assistant with helpful information to support you in navigating the world of caregiving for children with cancer. It has access to the Children's Oncology Group Family Handbook, a trusted resource for pediatric oncology information. Please give it a moment to set itself up.")

csv_filename = "Updated_English_COG_Family_Handbook.csv"
retriever = loading_handbook(csv_filename)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-4o", openai_api_key=openai_api_key, temperature=0, streaming=True, verbose=True
)

custom_template = """
You are a cancer care assistant with access to the Children's Oncology Group Family Handbook. 
Please assist the user with their query, and keep the answer based on the document you have access to. 
Be empathetic and provide accurate information.

Chat History:
{chat_history}

Question: 
{question}
"""
prompt = PromptTemplate.from_template(custom_template)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, chain_type="stuff",
    condense_question_prompt=prompt
)

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    # Process the LLM response
    retrieval_handler = PrintRetrievalHandler(st.container())
    stream_handler = StreamHandler(st.empty())
    
    response = qa_chain.run(
        {"chat_history": msgs.messages, "question": user_query}, 
        callbacks=[retrieval_handler, stream_handler]
    )

    # Save the conversation in Firestore (no user ID needed)
    save_conversation_to_firestore(user_query, response, retrieval_handler.headings)
