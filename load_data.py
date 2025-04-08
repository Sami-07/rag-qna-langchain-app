from db import collection, client, dbName, collectionName
from pymongo import MongoClient
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import os
from dotenv import load_dotenv
load_dotenv()

print("Starting to load files...")
data = []
for filename in os.listdir("./sample_files"):
    if filename.endswith(".txt"):
        print(f"Loading {filename}...")
        loader = TextLoader(f"./sample_files/{filename}")
        data.extend(loader.load())
print(f"Loaded {len(data)} documents in total")

print("Creating embeddings...")
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

print("Storing documents in vector store...")
vectorStore = MongoDBAtlasVectorSearch.from_documents(
    data,
    embeddings,
    collection=collection,
)
print("Process completed!")
