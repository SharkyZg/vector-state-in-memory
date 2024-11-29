import os

import dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

dotenv.load_dotenv()

if __name__ == "__main__":
    pdf_path = "./NIPS-2017-attention-is-all-you-need-Paper.pdf"
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    chunks = text_splitter.split_documents(docs)
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

    query ="explain attention mechanism in transformer models like I'm farmer 107 years old."

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    if not os.path.exists("faiss_index_react"):
        vector_store.save_local("faiss_index_react")

    new_vector_store = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

    combine_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_documents_chain)

    result = retrieval_chain.invoke({"input": query})
    print(result["answer"])
