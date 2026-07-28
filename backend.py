import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import (
    OpenAIEmbeddings,
    ChatOpenAI
)

from langchain_community.vectorstores import FAISS

from langchain.chains import RetrievalQA

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------
# LLM
# ----------------------------------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# ----------------------------------------------------
# Create Vector Database
# ----------------------------------------------------

def create_vector_db(pdf_path):

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    vector_db = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_db


# ----------------------------------------------------
# Ask Question
# ----------------------------------------------------

def ask_question(question, vector_db):

    qa = RetrievalQA.from_chain_type(

        llm=llm,

        chain_type="stuff",

        retriever=vector_db.as_retriever(

            search_kwargs={
                "k":4
            }

        ),

        return_source_documents=True
    )

    result = qa.invoke(
        {
            "query": question
        }
    )

    answer = result["result"]

    sources = result["source_documents"]

    return answer, sources