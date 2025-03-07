from gradio.components import Label
from numpy import ediff1d
import torch
import time
from transformers import pipeline,AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import transformers
from langchain_openai import OpenAI,OpenAIEmbeddings
from langchain_community.llms import CTransformers
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chat_models import init_chat_model
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough


from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Function to transcribe audio using the OpenAI Whisper model
def transcript_audio(audio_file):
    # Initialize the speech recognition pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
            )
    # Transcribe the audio file and return the result
    result = pipe(audio_file, batch_size=8)["text"]

    text_spillter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 0)
    chunked_docs = text_spillter.split_text(result)
    faiss(chunked_docs)
    return chunked_docs

def faiss(chunked_text):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(chunked_text,embeddings)
    db.save_local("faiss_index_data")



def answer_question(question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("faiss_index_data",embeddings,allow_dangerous_deserialization=True)

    model = init_chat_model("llama3-8b-8192", model_provider="groq")

    prompt_template = """
    ### [INST]
    Instruction: Answer the question First try to find from context given below and if you are not able to do
    so then anser question based on your knowlwdge and don't include words like 'based on context' or 'according to context' or 'based on my knowledge'
    in final answer. Here is context to help:

    {context}

    ### QUESTION:
    {question}

    [/INST]
     """

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    chain = prompt | model

    retriever = db.as_retriever()

    rag_chain = ({"context":retriever,"question":RunnablePassthrough()} | chain)
    response = rag_chain.invoke(question)

    # time = str(time.time())
    # path = "/data/outpua_data/output.wav"
    # sf.write(path, speech.numpy(), samplerate=16000)

    return response.content
