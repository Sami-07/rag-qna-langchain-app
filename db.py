from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

client = MongoClient(os.getenv("MONGO_URI"))

dbName = "langchain_rag"
collectionName = "rag_collection_of_documents"
collection = client[dbName][collectionName]

__all__ = ["collection", "client", "dbName", "collectionName"]