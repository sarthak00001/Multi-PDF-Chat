# pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub

import streamlit as st
from dotenv import load_dotenv  ## to acces the .env file
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
import sentence_transformers
from htmlTemplates import css,bot_template,user_template
# from langchain_community.llms import HuggingFaceEndpoint, HuggingFaceHub
# from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceHub

from langchain_community.chat_models import ChatHuggingFace

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline


def get_pdf_text(pdf_docs):
    text = ""                               ## Contains all the raw text from pdf
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)         ## Create PDF object that has pages and from this pages will be able to read text
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,                ## So that everytext is in the chunks and nothing is left behind
        length_function = len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):           ## vector DB to store the chunks for reading the data
    # embeddings = OpenAIEmbeddings()       ## using OPENAI (its paid)

    # embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")      # is very large (5GB) hence using the other small onle 
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    # embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()                                                                              ## OpenAI paid
    # llm = HuggingFaceHub(repo_id = "HuggingFaceH4/zephyr-7b-beta", task="text-generation", model_kwargs = {"temperature":0.5, "max_length":512})

    # base_llm = HuggingFaceEndpoint(
    #     repo_id="google/flan-t5-base",
    #     task="text2text-generation",
    #     temperature=0.5,                      #controls randomness; lower = more deterministic, higher = more creative.
    #     max_new_tokens=512,
    #     provider =  'arxiv'
    # )                                       
    # llm = ChatHuggingFace(llm=base_llm)

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    llm = HuggingFacePipeline(pipeline = pipe)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)              ## For memmory of chatbot
    conversation_chain = ConversationalRetrievalChain.from_llm(                                     ## Allows to chat with our vector DB and have some memory to it
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory,
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    # st.write(response)
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon =":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs header :books:")
    user_question = st.text_input("Ask a question about your documents:")

    if user_question:
        handle_user_input(user_question)

    # st.write(user_template.replace("{{MSG}}","HELLLLUUUU"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}","Hello"),unsafe_allow_html=True)

    with st.sidebar:      ##Everything after this will be stored in the sidebar
        st.subheader("Your documents to be read")
        pdf_docs = st.file_uploader("Upload you PDFs here and click on 'Proces'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)                    # To see if its really working and display all the text

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                # Create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain (# session_state -> to not reload the variable again when any button is hit (like strealit does))
                st.session_state.conversation = get_conversation_chain(vectorstore) # takes the history of conv and return the next element in conv 



if __name__ == '__main__':
    main()