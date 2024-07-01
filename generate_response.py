from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from encode_corpus import collection, tokenizer, model
from retrieve import retrieve, queries

gen_model_name = "facebook/bart-large-cnn"
gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)


def generate_response(query, retrieved_docs, tokenizer, model):
    context = "".join(retrieved_docs)
    input_text = f"Query: {query}"
    # print("Input text:", input_text)
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(
        inputs.input_ids, max_length=150, num_beams=2, early_stopping=True
    )
    response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return response


for query in queries:
    retrieved_docs = retrieve(query, collection, tokenizer, model)
    response = generate_response(query, retrieved_docs, gen_tokenizer, gen_model)
    print(f"Query: {query}")
    print("Response:", response)
    print()
