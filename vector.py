from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

df = pd.read_csv("mental_health_qa.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chroma_mentalhealth_db"
add_documents = not os.path.exists(db_location)

vector_store = Chroma(
    collection_name="mental_health_qa",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    documents = []
    ids = []
    for i, row in df.iterrows():
        document = Document(
            page_content=f"Q: {row['Questions']}\nA: {row['Answers']}",
            metadata={"Question_ID": row["Question_ID"]},
            id=str(row["Question_ID"])
        )
        ids.append(str(row["Question_ID"]))
        documents.append(document)
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})
