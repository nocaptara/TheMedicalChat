#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!pip install streamlit


# In[5]:


#!pip install streamlit_chat


# In[13]:


#!pip install langchain


# In[31]:


import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

# In[9]:


#load the pdf files from the path
loader = DirectoryLoader("C:/Users/tarang rathod/NLP/medical_chatbot(new)/data/",glob="*.pdf",loader_cls=PyPDFLoader)
documents = loader.load()


# In[10]:


#split text into chunks
text_splitter  = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)


# In[11]:


#create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                   model_kwargs={'device':"cpu"})


# In[12]:


#vectorstore
vector_store = FAISS.from_documents(text_chunks,embeddings)


# In[14]:


#!pip install torch


# In[15]:


#!pip install accelerate


# In[16]:


#!pip install bitsandbytes


# In[17]:


#!pip install transformers


# In[18]:


#!pip install sentence_transformers


# In[19]:


#!pip install streamlit


# In[20]:


#!pip install streamlit_chat


# In[21]:


#!pip install faiss-cpu


# In[22]:


#!pip install altair


# In[23]:


#!pip install tiktoken


# In[24]:


#!pip install huggingface-hub


# In[39]:


text_chunks


# In[27]:


vector_store = FAISS.from_documents(text_chunks,embeddings)


# In[33]:


from langchain.llms import CTransformers


# In[35]:


#create llm
llm = CTransformers(model="C:/Users/tarang rathod/NLP/medical_chatbot(new)/model/llama-2-7b-chat.ggmlv3.q4_0.bin",model_type="llama",
                    config={'max_new_tokens':128,'temperature':0.01})


# In[36]:


memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


# In[30]:


#!pip install ctransformers


# In[37]:


chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)


# In[38]:


st.title("HealthCare ChatBot 🧑🏽‍⚕️")

def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()