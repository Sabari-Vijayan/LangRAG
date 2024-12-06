import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables (ensures OpenAI API key is available)
load_dotenv()

# Constants for configuration
CHROMA_PATH = "chroma"  # Directory where vector database is stored
PROMPT_TEMPLATE = """
You are a helpful AI assistant tasked with answering questions based strictly on the provided context.

Context:
{context}

---
Question: {question}

Please follow these guidelines:
1. Base your answer only on the information in the given context
2. If the context does not contain enough information to answer, say "I cannot find a definitive answer in the provided documents."
3. Be concise but thorough
4. Cite the source of information if possible
"""

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Query your document database")
    parser.add_argument("query_text", type=str, help="The query text to search in your documents")
    parser.add_argument("--k", type=int, default=3, help="Number of top similar documents to retrieve")
    parser.add_argument("--threshold", type=float, default=0.7, help="Relevance threshold for document matching")
    args = parser.parse_args()

    # Initialize embedding function using OpenAI
    try:
        embedding_function = OpenAIEmbeddings()
    except Exception as e:
        print(f"Error initializing embeddings: {e}")
        print("Ensure your OpenAI API key is correctly set in the .env file")
        return

    # Load the existing Chroma vector database
    try:
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=embedding_function
        )
    except Exception as e:
        print(f"Error loading vector database: {e}")
        print(f"Ensure the Chroma database exists at {CHROMA_PATH}")
        return

    # Perform similarity search
    try:
        results = db.similarity_search_with_relevance_scores(
            args.query_text, 
            k=args.k
        )
    except Exception as e:
        print(f"Error performing similarity search: {e}")
        return

    # Check if relevant results were found
    if len(results) == 0 or results[0][1] < args.threshold:
        print(f"Unable to find matching results with relevance above {args.threshold}")
        return

    # Prepare context by concatenating document contents
    context_text = "\n\n---\n\n".join([
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content}" 
        for doc, score in results
    ])

    # Create prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    # Format the prompt with context and query
    prompt = prompt_template.format(
        context=context_text, 
        question=args.query_text
    )

    # Initialize ChatOpenAI model
    try:
        model = ChatOpenAI(
            model='gpt-3.5-turbo',  # You can change the model as needed
            temperature=0.1  # Low temperature for more deterministic responses
        )
    except Exception as e:
        print(f"Error initializing ChatOpenAI model: {e}")
        return

    # Generate response
    try:
        response_text = model.predict(prompt)
    except Exception as e:
        print(f"Error generating response: {e}")
        return

    # Prepare and print the final response with sources
    sources = [
        f"{doc.metadata.get('source', 'Unknown')} (Relevance: {score:.2f})" 
        for doc, score in results
    ]

    # Formatted output with response and sources
    print("\n--- AI Response ---")
    print(response_text)
    print("\n--- Sources ---")
    for source in sources:
        print(source)

if __name__ == "__main__":
    main()