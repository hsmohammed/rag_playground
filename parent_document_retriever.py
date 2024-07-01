from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


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

# print(docs)
# print(type(docs))
# print(len(docs))
# print(docs[0])
documents = [
    Document(page_content=doc["content"], metadata=doc["meta"]) for doc in docs
]


child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# # Huggingface embedding
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

vectorstore = Chroma(
    collection_name="full_documents",
    embedding_function=hf_embeddings,
)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
)

retriever.add_documents(documents)

# print(list(store.yield_keys()))
sub_docs = vectorstore.similarity_search("justice breyer")
print(len(sub_docs))
print(len(sub_docs[0].page_content))
print(sub_docs[0].metadata)

retrieved_docs = retriever.invoke("justice breyer")
print(len(retrieved_docs))
print(len(retrieved_docs[0].page_content))

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

vectorstore = Chroma(
    collection_name="split_parents",
    embedding_function=hf_embeddings,
)
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(documents)


sub_docs = vectorstore.similarity_search("justice breyer")
print(len(sub_docs))

retrieved_docs = retriever.invoke("justice breyer")
print(len(retrieved_docs))
print(len(retrieved_docs[0].page_content))
