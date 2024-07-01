from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import SearchType
import uuid
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline


class CustomTextLoader:
    def __init__(self, file_path, encoding="utf-8"):
        self.file_path = file_path
        self.encoding = encoding

    def load(self):
        with open(self.file_path, "r", encoding=self.encoding) as f:
            text = f.read()
        return [{"content": text, "meta": {"source": self.file_path}}]


loaders = [
    CustomTextLoader("text/paul_graham_essay.txt"),
    CustomTextLoader("text/state_of_the_union.txt"),
]
docs = []

for loader in loaders:
    docs.extend(loader.load())


documents = [
    Document(page_content=doc["content"], metadata=doc["meta"]) for doc in docs
]


text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000)
docs = text_splitter.split_documents(documents)


print(len(docs))

# # Huggingface embedding
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
# The vectorstore to use to index the child chunks
vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=hf_embeddings,
)
# The storage layer for the parent documents
store = InMemoryByteStore()
id_key = "doc_id"
# The retriever (empty to start)
retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    byte_store=store,
    id_key=id_key,
)

doc_ids = [str(uuid.uuid4()) for _ in docs]
print(len(doc_ids))


# The splitter to use to create smaller chunks
child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
sub_docs = []
for i, doc in enumerate(docs):
    _id = doc_ids[i]
    _sub_docs = child_text_splitter.split_documents([doc])
    for _doc in _sub_docs:
        _doc.metadata[id_key] = _id
    sub_docs.extend(_sub_docs)

# print(len(sub_docs))
# print(sub_docs[0])

retriever.vectorstore.add_documents(sub_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))

res = retriever.vectorstore.similarity_search("justice breyer")
print(len(res))


# create summaries of docs with huggingface transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "t5-small"  # Using a smaller model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def summarize_document(document, max_length=150, min_length=30):
    inputs = tokenizer.encode(
        "summarize: " + document, return_tensors="pt", max_length=512, truncation=True
    )
    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


summaries = []
for doc in docs:
    summaries.append(summarize_document(doc.page_content))

summary_docs = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(summaries)
]

retriever.vectorstore.add_documents(summary_docs)
retriever.docstore.mset(list(zip(doc_ids, docs)))


sub_docs = vectorstore.similarity_search("justice breyer")

print(sub_docs[0])
# print(type(summaries))
retrieved_docs = retriever.invoke("justice breyer")

print(len(retrieved_docs[0].page_content))
