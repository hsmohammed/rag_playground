from transformers import AutoTokenizer, AutoModel
import torch
import chromadb

client = chromadb.Client()
collection_name = "rag_demo"
collection = client.create_collection(name=collection_name)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def encode_text(texts, tokenizer, model):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.numpy().tolist()


corpus = [
    "The capital of France is Paris.",
    "The Eiffel tower is one of the most famous landmarks in the Paris.",
    "Paris is known for its cafe culture and landmarks like the Nore Dame Cathedral.",
    "The Louvre is the worlde's largest museum and a historic monument in Paris.",
]

corpus_embeddings = encode_text(corpus, tokenizer, model)

for i, embedding in enumerate(corpus_embeddings):
    collection.add(
        str(i),
        embedding,
        metadatas={"text": corpus[i]},
    )
