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
import csv


openai_api_key = st.secrets["andrew_openai_api_key"]
st.title('👪  Caregiving Handbook')
st.markdown('🙏🏼 Welcome to the Handbook Healthcare Assistant! We have provided the assistant with helpful information to support you in navigating the world of caregiving for children with cancer. It has access to the Children\'s Oncology Group Family Handbook, a trusted resource for pediatric oncology information. Please give it a moment to set itself up.')

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
        self.initial_prompt = ""  # Add an attribute to store the initial prompt


    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts and len(prompts) > 0:
            self.initial_prompt = prompts[0]  # Store the initial prompt
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")
        # print("Initial LLM Prompt:", self.initial_prompt)

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        # print("New token from LLM:", token)
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Retrieving Handbook Information**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        # Display initial retrieving context message
        self.final_status.text("Retrieving context for your question...")

    def on_retriever_end(self, documents, **kwargs):
        content = "**Handbook Links:**\n\n"
        for idx, doc in enumerate(documents):
            # Assuming doc is an instance of Document and has metadata as a dictionary
            # Directly access attributes of the Document object
            page_number = doc.metadata['labeled_page_number'] if 'labeled_page_number' in doc.metadata else 'Unknown Page Number'
            page_header = doc.metadata['page_header'] if 'page_header' in doc.metadata else 'Unknown Header'
            page_link = doc.metadata['page_link']
            source = f"[{page_header}, Page {page_number}]({page_link})"  
            
            content += f"**Document {idx + 1} from {source}:**\n{doc.page_content}\n\n"
            self.status.markdown(source)
        # Update the status container with the final content
        self.status.update(state="complete")


csv_filename = "Updated_English_COG_Family_Handbook.csv"
retriever = loading_handbook(csv_filename)

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview", openai_api_key=openai_api_key, temperature=0, streaming=True, verbose=True
)

custom_template = """
You are a cancer care assistant with access to the Children's Oncology Group Family Handbook. 
Please assist the user with their query, and keep the answer based on the document you have access to the 
The Childrens Oncology Group Family Handbook. Be empathetic and provide accurate information.

Chat History:
{chat_history}

Question: 
{question}
"""


qa_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever=retriever, memory=memory, verbose=True, 
    # condense_question_prompt=PromptTemplate.from_template(custom_template)
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

system_prompt="""
System Prompt: 

Always include citations to the orginal material. Keep the answer about one paragraph long.
- Include inline citations in square brackets within the paragraph.
- List all references at the end under 'References' with the format: 'Most relevant heading, **Page Header, Page Number**'.

Example:
Human : "What is the treatment for cancer?"
Assistant:   The treatment for leukemia includes chemotherapy and radiation therapy [1].The treatment for ALL includes chemotherapy and radiation therapy [2].
        References:
        1. Leukemia Treatment Overview, **Treatment Procedures**, Page 23"
        2. ALL Treatment Overview, **Treatment Procedures**, Page 36"

"""



if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(system_prompt+user_query, callbacks=[retrieval_handler, stream_handler])