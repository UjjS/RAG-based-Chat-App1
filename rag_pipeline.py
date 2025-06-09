import os
import streamlit as st
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Custom prompt banaya taki LLM jab answer na ho toh clearly bol de ki nahi pata
CUSTOM_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Chat History: {chat_history}
Question: {question}
Helpful Answer:"""

def get_vector_store(text_chunks, openai_api_key: str):
    """
    Text chunks se Qdrant vector store banata hai.
    Streamlit cache use karta hai taki bar bar vector store na banana pade.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    
    # Unique collection name banaya, first chunk ke hash se
    # Isse app restart hone par bhi wahi collection reuse ho sakti hai
    collection_name = f"collection-{hash(text_chunks[0])}"

    vectorstore = Qdrant.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        url=os.getenv("QDRANT_URL", "http://qdrant:6333"), # Docker service ka naam diya
        collection_name=collection_name,
    )
    return vectorstore


def get_conversation_chain(vector_store, openai_api_key: str):
    """
    Conversational retrieval chain banata hai.
    Ye chain vector store se relevant text nikalta hai aur LLM se response generate karwata hai.
    """
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0.3,
        openai_api_key=openai_api_key
    )
    
    # Conversation ka pura history yaad rakhne ke liye memory use kiya
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )

    # Documents combine karne ke liye prompt banaya
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=CUSTOM_PROMPT_TEMPLATE
    )

    # Main conversational chain yahan ban rahi hai
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return conversation_chain