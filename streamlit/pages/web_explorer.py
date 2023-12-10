import logging

import faiss
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import FAISS

import streamlit as st

load_dotenv()
st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")


def settings():
    # Vectorstore
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(
        embeddings_model.embed_query, index, InMemoryDocstore({}), {}
    )

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0, streaming=True)

    # Search
    search = GoogleSearchAPIWrapper()

    # Initialize
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public, llm=llm, search=search, num_search_results=3
    )

    return web_retriever, llm


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.header("`Interweb Explorer`")
st.info(
    "`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different modes: public API or private (no data sharing).`"
)

# Make retriever and llm
if "retriever" not in st.session_state:
    st.session_state["retriever"], st.session_state["llm"] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input
question = st.text_input("`Ask a question:`")

if (len(question) > 0) & (st.button("検索実行")):
    # Generate answer (w/ citations)
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain(
        {"question": question}, callbacks=[retrieval_streamer_cb, stream_handler]
    )
    st.session_state["result"] = result
if "result" in st.session_state:
    result = st.session_state["result"]
    st.info("`Answer:`\n\n" + result["answer"])
    st.info("`Sources:`\n\n" + result["sources"])
