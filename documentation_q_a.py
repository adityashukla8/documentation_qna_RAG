# {{{ imports

import pandas as pd
import numpy as np

from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.utilities import ApifyWrapper
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
# from langchain_voyageai import VoyageAIEmbeddings

from InstructorEmbedding import INSTRUCTOR

import faiss
import pickle

import matplotlib.pyplot as plt
from functools import partial

import os

from  langchain.schema import Document
import json
from typing import Iterable

# }}}
# {{{ keys

APIFY_API_TOKEN = os.environ.get('APIFY_API_TOKEN')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
VOYAGE_API_KEY = os.environ.get('VOYAGE_API_KEY')

# }}}
# {{{ ApifyWrapper

# apify = ApifyWrapper()
# loader = apify.call_actor(
#     actor_id='apify/website-content-crawler',
#     run_input={'startUrls':[{'url': 'https://docs.llamaindex.ai/en/stable/module_guides/indexing/'}]},
#     dataset_mapping_function=lambda item: Document(page_content=item["text"] or "", metadata={"source": item["url"]}), 
# )

# docs = loader.load()
# print(len(docs))
# }}}
# {{{ def save_docs_to_josn

def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')

# }}}
# {{{ def load_docs_from_json

def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

# }}}
# {{{ load and analyze docs

#save_docs_to_jsonl(docs, 'llamaindex_docs.jsonl')
docs = load_docs_from_jsonl("/home/adityashukla/Aditya Shukla/LLM/llamaindex_docs.jsonl")
docs_length = []
for i in range(len(docs)):
    docs_length.append(len(docs[i].page_content))

print(f'doc lengths\nmin: {min(docs_length)} \navg.: {round(np.average(docs_length), 1)} \nmax: {max(docs_length)}')
# }}}
# {{{ plot chunks

#each doc has variable chunks of characters in it
# plt.figure(figsize=(15, 3))
# plt.plot(docs_length, marker='o')
# plt.title("doc length")
# plt.ylabel("# of characters")
# plt.show()

# }}}
# {{{ initialize chunks and text_splitter

chunk_size = 1000
chunk_overlap = 200
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len)

# }}}
# {{{ def chunk_docs

# sample_doc = docs[0]
# chunks = text_splitter.create_documents(texts=[sample_doc.page_content], metadatas=[{'source': sample_doc.metadata['source']}])
# print(f'Sample doc (one doc) split into {len(chunks)} chunks.')
# chunks[0]

def chunk_docs(doc, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len)
    chunks = text_splitter.create_documents(texts=[doc.page_content], metadatas=[{'source': doc.metadata['source']}])
    return chunks

# }}}
# {{{ flatten chunks

chunked_docs = []

for i in docs:
    chunked_docs.append(chunk_docs(i, chunk_size=chunk_size, chunk_overlap=chunk_overlap))
flattened_chunked_docs = [doc for docs in chunked_docs for doc in docs]
# }}}
# {{{ def store_embeddings 

def store_embeddings(docs, embeddings, store_name, path):
    vectorstore = FAISS.from_documents(docs, embeddings)

    with open(f'{path}/faiss_{store_name}.pkl', 'wb') as f:
        pickle.dump(vectorstore, f)

# }}}
# {{{ def load_embeddings

def load_embeddings(store_name, path):
    with open(f'{path}/faiss_{store_name}.pkl', 'rb') as f:
        VectorStore = pickle.load(f)
    return VectorStore

# }}}
# {{{ load thenlper/gte-base embedding model and retriver

# instructor_embeddings = HuggingFaceInstructEmbeddings(model_name='thenlper/gte-base', model_kwargs={'device': 'cpu'})
Embedding_store_path = f'/home/adityashukla/Aditya Shukla/LLM/embedding_store/'
#store_embeddings(flattened_chunked_docs[:15000], instructor_embeddings, store_name='instructEmbeddings', path=Embedding_store_path)
db_instructEmbedding = load_embeddings('instructEmbeddings', Embedding_store_path)
retriver = db_instructEmbedding.as_retriever(search_type='similarity', search_kwargs={'k':10})
#qa_docs = retriver.get_relevant_documents('What types of data can LlamaIndex index')

# }}}
# {{{ qa_chain-instructembed

qa_chain_instructembed = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0.2,), 
    chain_type='stuff', 
    retriever=retriver, 
    return_source_documents=True, 
    verbose=False)

# }}}
# {{{ openai embeddings

#OPENAI embeddings
# openai_embeddings = OpenAIEmbeddings()
#store_embeddings(flattened_chunked_docs[:1500], openai_embeddings, store_name='openaiEmbeddings', path=Embedding_store_path)
# db_openai_embeddings= FAISS.from_documents(flattened_chunked_docs, openai_embeddings)
# retriver_openai = openai_embeddings.as_retriever(search_type='similarity', search_kwargs={'k':5})

# }}}
# {{{ qa_chain openai

# qa_chain_openai = RetrievalQA.from_chain_type(llm=OpenAI(temperature=0.2,), 
#     chain_type='stuff', 
#     retriever=retriver, 
#     return_source_documents=True, 
#     verbose=False)

# }}}
# {{{ load hkunlp/instructor-large embedding model and retriver

# instructor_embeddings_large = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-large', model_kwargs={'device': 'cpu'})
# store_embeddings(flattened_chunked_docs[:15000], instructor_embeddings_large, store_name='instructEmbeddings_hkunlp', path=Embedding_store_path)
# retriver = db_instructEmbedding.as_retriever(search_type='similarity', search_kwargs={'k':5})

# }}} 

query = 'how to index data, share code example as well'
response = qa_chain_instructembed(query)
print(f'\n\nanswer:\n',  f'{response["result"]}', '\n\nsource: \n', f'{[source.metadata["source"] for source in response["source_documents"]]}')

