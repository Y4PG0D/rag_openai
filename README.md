# rag_openai

This is a simple rag pipeline used to get information on a topic based on specific context using a simple rag model. By connecting to the OpenAI and Pinecone APIs, a query is sent to OpenAI together with context in the form of a txt file. THE OpenAPI is then prompted to answer the query only from the provided context. If no answer for the specific query can be found in the context, the model is prompted to answer with "I don't know"

# Requirements

To run the script you need the following:

1. Create .venv as your virtual envrionement.
2. Install the required packages using "pip install -r requirements.txt"
3. Create .env file and add your OpenAI and Pinecone API Secret Keys:
    OPENAI_API_KEY=your_key n/
    PINECONE_API_KEY=your key