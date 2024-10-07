import streamlit as st
from os import environ
from typing import List
import chromadb
import pandas as pd
from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.docstore.document import Document
from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
from transformers import AutoTokenizer

load_dotenv()

# Constants
EMBED_MODEL_NAME = "jina-embeddings-v2-base-en"
LLM_NAME = "mixtral-8x7b-32768"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 8192
CSV_FILE_PATH = "./file/data.csv"
VECTOR_STORE_DIR = "./vectorstore/"
COLLECTION_NAME = "collection1"

# Configure logging
logging.basicConfig(
    filename='application.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_csv_data() -> List[Document]:
    """Loads the CSV file and converts it to a list of Documents."""
    try:
        st.info("[+] Loading CSV data...")
        df = pd.read_csv(CSV_FILE_PATH)
        st.info(f"[+] CSV data loaded, total rows: {len(df)}")

        documents = []
        for _, row in df.iterrows():
            content = " ".join([f"{col}: {row[col]}" for col in df.columns])
            documents.append(Document(page_content=content))

        return documents
    except Exception as e:
        st.error(f"[-] Error loading the CSV file: {e}")
        return []


def chunk_document(documents: List[Document]) -> List[Document]:
    """Splits the input documents into maximum of CHUNK_SIZE chunks."""
    tokenizer = AutoTokenizer.from_pretrained(
        "jinaai/" + EMBED_MODEL_NAME, cache_dir=environ.get("HF_HOME")
    )
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_SIZE // 50,
    )

    st.info(f"[+] Splitting documents...")
    chunks = text_splitter.split_documents(documents)
    st.info(f"[+] Document splitting done, {len(chunks)} chunks total.")

    return chunks


def create_and_store_embeddings(embedding_model: JinaEmbeddings, chunks: List[Document]) -> Chroma:
    """Calculates the embeddings and stores them in a Chroma vectorstore."""
    vectorstore = Chroma(
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
    )

    batch_size = 166  # Chroma's limit on batch size
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch_chunks = chunks[i: i + batch_size]
        ids = [str(i + j) for j in range(len(batch_chunks))]
        metadatas = [doc.metadata for doc in batch_chunks]
        embeddings = embedding_model.embed_documents([doc.page_content for doc in batch_chunks])

        vectorstore._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )

    st.info("[+] Vectorstore created with batched embeddings.")
    return vectorstore


def get_vectorstore_retriever(embedding_model: JinaEmbeddings) -> VectorStoreRetriever:
    """Returns the vectorstore."""
    try:
        vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name=COLLECTION_NAME,
            persist_directory=VECTOR_STORE_DIR,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.info("[+] Vectorstore loaded successfully.")
        return retriever
    except Exception as e:
        st.warning(f"[-] Collection not found: {e}. Creating a new one.")
        csv_data = load_csv_data()
        if not csv_data:
            st.error("[-] No data available to create a vectorstore.")
            return None
        chunks = chunk_document(csv_data)
        return create_and_store_embeddings(embedding_model, chunks).as_retriever(search_kwargs={"k": 3})


def create_rag_chain(embedding_model: JinaEmbeddings, llm: ChatGroq) -> Runnable:
    """Creates the RAG chain for course recommendations based on user input."""
    template = """Based on the user's learning preferences, 
    The user might describe their level of expertise or specific subjects they are interested in.

    <context>
    {context}
    </context>

    User Input: {input}

    Please suggest suitable courses for the user, or inform them if no relevant data is available.
    If no courses match the input, respond with: "Data not available for this."
    """
    prompt = ChatPromptTemplate.from_template(template)

    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    retriever = get_vectorstore_retriever(embedding_model)

    if retriever is None:
        raise ValueError("[-] Retriever could not be created.")

    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain


def run_chain(chain: Runnable, query: str) -> str:
    """Run the RAG chain with the user query."""
    response = chain.invoke({"input": query})
    context = response["context"]
    answer = response["answer"]

    st.markdown(f"**Context:**")
    for doc in context:
        st.markdown(f"- {doc.metadata} | {doc.page_content[:50]}...")

    return answer


def main() -> None:
    st.title("Analytical Vidya Course Recommendations")

    embedding_model = JinaEmbeddings(
        jina_api_key=environ.get("JINA_API_KEY"),
        model_name=EMBED_MODEL_NAME,
    )

    llm = ChatGroq(temperature=LLM_TEMPERATURE, model_name=LLM_NAME)

    chain = create_rag_chain(embedding_model=embedding_model, llm=llm)

    query = st.text_input("Enter your learning preferences:")
    if st.button("Get Recommendations"):
        if query:
            st.info(f"Processing query: {query}")
            answer = run_chain(chain, query)
            st.markdown(f"**Recommendation:** {answer}")
        else:
            st.error("Please enter a valid query.")


if __name__ == "__main__":
    main()
