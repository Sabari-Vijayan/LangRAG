import argparse
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline

# Define a custom embedding class using Hugging Face Sentence Transformers
class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list):
        return [self.model.encode(text) for text in texts]

    def embed_query(self, text: str):
        return self.model.encode(text)

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB with Hugging Face embeddings.
    embedding_function = HuggingFaceEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Use a Hugging Face model for response generation (optional).
    generator = pipeline("text-generation", model="gpt2")  # You can use other models like GPT-3/Neo

    # Then use the generator to produce a response:
    response = generator(prompt, max_new_tokens=100, num_return_sequences=1)
    response_text = response[0]["generated_text"]

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()