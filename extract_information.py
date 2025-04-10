from db import collection
from pymongo import MongoClient
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import os
from dotenv import load_dotenv
from pydantic import SecretStr
load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set")

api_key_secret = SecretStr(OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(api_key=api_key_secret)

vectorStore = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collection,
    index_name="langchain_index"
)


def query_data(query):
    print("query: ", query)
    print("vectorStore: ", vectorStore.embeddings)
    docs = vectorStore.similarity_search(query, k=1)
    print("docs: ", docs)
    as_output = docs[0].page_content

    llm = OpenAI(api_key=api_key_secret)
    retriever = vectorStore.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever
    )
    retriever_output = qa.run(as_output)
    return as_output, retriever_output


def main():
    with gr.Blocks() as demo:
        gr.Markdown("Extract Information from Text")
        query = gr.Textbox(placeholder="Enter your query")
        as_output = gr.Textbox()
        retriever_output = gr.Textbox()
        submit_button = gr.Button("Submit")

        submit_button.click(query_data, inputs=query, outputs=[as_output, retriever_output])
    
    demo.launch()



if __name__ == "__main__":
    main()





