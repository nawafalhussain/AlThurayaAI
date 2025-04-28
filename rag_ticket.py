from langchain_community.vectorstores.pgvector import PGVector 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate




CONNECTION_STRING = "postgresql://nawafalhussain:postgres@localhost:5432/rag_db"
COLLECTION_NAME = "ticket"
file_path = "/Users/nawafalhussain/Downloads/user_simple_guide_password_printer.pdf"



def ingest_pdf(file_path: str):
    from langchain_community.vectorstores.pgvector import PGVector
    from langchain_community.embeddings import OllamaEmbeddings

    try:
        print(f"📄 Loading PDF: {file_path}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.split_documents(documents)
        
        print(f"🧩 Number of chunks: {len(docs)}")
        print(f"📄 First chunk: {docs[0].page_content[:200]}...")



        embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            encode_kwargs={'normalize_embeddings': True}
)
        # Use the correct initialization method
        PGVector.from_documents(
            documents=docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING
        )

        print(f"✅ Successfully ingested: {file_path}")

    except Exception as e:
        print(f"❌ Error ingesting {file_path}: {str(e)}")

def get_rag_chain():
    from langchain_community.vectorstores.pgvector import PGVector
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain.chains import RetrievalQA
    from langchain_community.llms import Ollama

    llm = Ollama(model="llama3")
    embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            encode_kwargs={'normalize_embeddings': True}
)

    vectorstore = PGVector(
        connection_string=CONNECTION_STRING,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever()
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
you are help desk assistance, i must replay to the users with the correct answer and more details and please support arabic language and english language. 
{context}
Question: {question}
Answer:
"""
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt}
)


def answer_from_ticket(question: str):
    qa = get_rag_chain()
    result = qa.invoke({"query": question})
    return qa.invoke(question)

if __name__ == "__main__":
    # Optional: for manual ingestion/testing
    ingest_pdf(file_path)


