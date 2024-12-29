import os
from dotenv import load_dotenv
import requests
import json
from pinecone import Pinecone, ServerlessSpec
import time

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Ensure you have this in your .env

# Pinecone setup
pinecone = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-qa-bot"

if index_name not in pinecone.list_indexes().names():
    pinecone.create_index(
        name=index_name,
        dimension=768,  # Gemini embeddings are 768 dimensions
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pinecone.Index(index_name)

# Gemini Embedding Function
def get_gemini_embedding(text):
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/embedding-gecko-001:embedText?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json()['embedding']['value']
    except requests.exceptions.RequestException as e:
        print(f"Error getting embedding: {e}")
        return None

# Add Data to Pinecone (Batched)
def add_to_pinecone(documents, batch_size=5):
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        embeddings = []
        for text in batch:
            embedding = get_gemini_embedding(text)
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"Skipping document due to embedding error: {text}")

        if embeddings: # Check if embeddings were successfully generated
            upserts = [(str(i + j), embeddings[j]) for j in range(len(embeddings))]
            index.upsert(vectors=upserts)
            time.sleep(1) # Add a small delay between batches
        else:
            print("No embeddings generated for this batch. Skipping upsert.")

# Retrieval Function
def retrieve_similar_docs(query):
    query_embedding = get_gemini_embedding(query)
    if query_embedding:
        search_results = index.query(query_embedding, top_k=3, include_metadata=False)
        return [match.id for match in search_results.matches]
    else:
        print("Could not generate embedding for query.")
        return []

# Generate Answer with Gemini (Text Generation Model)
def generate_answer(query, context):
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/gemini-pro:generateText?key={GEMINI_API_KEY}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    data = {
        "prompt": {"text": prompt}
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()['candidates'][0]['output']
    except requests.exceptions.RequestException as e:
        print(f"Error generating answer: {e}")
        return None

# Example documents
documents = [
    "What is the refund policy?",
    "How can I contact customer support?",
    "What are the business hours?",
    "How to track my order?",
    "Do you offer international shipping?",
    "What payment methods do you accept?",
    "How do I create an account?",
    "What is your return address?",
    "Are there any discounts available?",
    "How do I reset my password?",
    "What is your warranty policy?",
    "Do you have a mobile app?",
    "How can I unsubscribe from emails?",
    "What are your terms of service?",
    "How do I delete my account?"
]

add_to_pinecone(documents)

# Test the QA Bot
query = "How can I get a refund?"
relevant_docs = retrieve_similar_docs(query)

if relevant_docs:
    context = "\n".join([documents[int(doc_id)] for doc_id in relevant_docs])
    answer = generate_answer(query, context)
    if answer:
        print(f"Q: {query}\nA: {answer}")
else:
    print("Could not retrieve relevant documents.")