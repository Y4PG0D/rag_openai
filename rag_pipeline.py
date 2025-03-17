import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


class RAGPipeline:
    def __init__(self, api_key_env="OPENAI_API_KEY", model_name="gpt-3.5-turbo", index_name="rag-test"):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv(api_key_env)

        # Initialize OpenAI model
        self.model = ChatOpenAI(openai_api_key=self.api_key, model=model_name)
        self.parser = StrOutputParser()

        # Create the prompt template
        self.template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
        """
        self.prompt = ChatPromptTemplate.from_template(self.template)

        # Load and process documents
        self.documents = self.load_and_split_documents("transcription.txt")

        # Create vector store
        self.embeddings = OpenAIEmbeddings()
        self.index_name = index_name
        self.pinecone = PineconeVectorStore.from_documents(self.documents, self.embeddings, index_name=self.index_name)

        # Define the retrieval chain
        self.chain = (
                {"context": self.pinecone.as_retriever(), "question": RunnablePassthrough()}
                | self.prompt
                | self.model
                | self.parser
        )

    def load_and_split_documents(self, file_path):
        loader = TextLoader(file_path)
        text_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        return text_splitter.split_documents(text_documents)

    def query(self, question):
        return self.chain.invoke(question)

    def similarity_search(self, query, top_k=3, similarity_metric="cosine"):
        return self.pinecone.similarity_search(query)[:top_k]


# Example usage:
if __name__ == "__main__":
    rag_pipeline = RAGPipeline()
    question = input("What is your question? ")
    response = rag_pipeline.query(question)
    print(response)