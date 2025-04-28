from langchain_community.vectorstores.pgvector import PGVector 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate




CONNECTION_STRING = "postgresql://nawafalhussain:postgres@localhost:5432/rag_db"
COLLECTION_NAME = "documents"
file_path = "/Users/nawafalhussain/Desktop/AI_Agent/help_desk/polices/Human_Resource_Policy_final_edition.pdf"



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
You are an HR Specialist assisting employees with their HR-related inquiries. Your goal is to provide clear, structured, and policy-compliant answers 
based on the company’s HR policies. Always reference the official HR policy document when applicable. If an inquiry is vague, ask for clarification before 
responding. If the information is not available, guide the employee to the appropriate HR contact or suggest alternative resources (e.g., intranet, manager). 
and please give me direct answer 
Maintain a professional yet friendly tone to ensure a positive employee experience. and please give me the answer direct📚 
"""
    )
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt}
)


def answer_from_rag(question: str):
    qa = get_rag_chain()
    return qa.invoke(question)

if __name__ == "__main__":
    # Optional: for manual ingestion/testing
    ingest_pdf(file_path)


