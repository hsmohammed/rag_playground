from langchain.retrievers import ParentDocumentRetriever

from langchain.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loaders = [
    TextLoader("text/paul_graham_essay.txt"),
    TextLoader("text/state_of_the_union.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())

# print(docs)
# print(type(docs))
# print(len(docs))
# print(docs[0])

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Huggingface embedding
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

retriever.add_documents(docs)

print(list(store.yield_keys()))
