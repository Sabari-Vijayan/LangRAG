from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.evaluation import load_evaluator

def main():
    # Use a popular, compact embedding model that works well for many tasks
    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",  # Lightweight, fast model
        model_kwargs={'device': 'cpu'},  # Specify device (CPU or CUDA)
        encode_kwargs={'normalize_embeddings': True}  # Normalize vectors
    )

    # Generate embedding for a word
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector[:10]}...")  # Print first 10 elements
    print(f"Vector length: {len(vector)}")

    # Compare semantic similarity
    evaluator = load_evaluator("pairwise_embedding_distance")
    words = ("apple", "iphone")
    similarity = evaluator.evaluate_string_pairs(
        prediction=words[0], 
        prediction_b=words[1]
    )
    print(f"Semantic distance between {words[0]} and {words[1]}: {similarity}")

if __name__ == "__main__":
    main()