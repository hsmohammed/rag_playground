from encode_corpus import encode_text, collection, tokenizer, model


def retrieve(query, collection, tokenizer, model, n_results=1):
    query_embedding = encode_text([query], tokenizer, model)[0]
    results = collection.query(query_embedding, n_results=n_results)
    return results["metadatas"][0][0]["text"]


queries = [
    "What is the capital of France?",
    "What is the most famous landmark in Paris?",
    "What is Paris known for?",
]

for query in queries:
    retrieved_docs = retrieve(query, collection, tokenizer, model)
    print(f"Query: {query}")
    print("Retrieved documents:", retrieved_docs)
    print()
