# from fastapi import FastAPI
# from langserve import add_routes
# from pirate_speak.chain import chain as pirate_speak_chain


# app = FastAPI()

# add_routes(app, pirate_speak_chain, path="/pirate-speak")


# @app.get("/")
# def root():
#     return {
#         "message": "Welcome to the langserve server! More information and routes can be found at /docs. 🦜 🏓",
#     }


# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}


# if __name__ == "__main__":
#     import uvicorn

#     uvicorn.run(app, host="0.0.0.0", port=8000)

#!/usr/bin/env python
"""Example LangChain server exposes a conversational retrieval chain.

Follow the reference here:

https://python.langchain.com/docs/expression_language/cookbook/retrieval#conversational-retrieval-chain

To run this example, you will need to install the following packages:
pip install langchain openai faiss-cpu tiktoken
"""  # noqa: F401

from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import format_document
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.vectorstores import FAISS

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

_TEMPLATE = """Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)

ANSWER_TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


import os
from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

load_dotenv()

anthropic_Api_Key: str = os.environ["ANTHROPIC_API_KEY"]

# _prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "Translate user input into pirate speak",
#         ),
#         ("human", "{text}"),
#     ]
# )
# _model = ChatAnthropic(anthropic_api_key=anthropic_Api_Key)

# # if you update this, you MUST also update ../pyproject.toml
# # with the new `tool.langserve.export_attr`
# chain = _prompt | _model


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

from langchain_openai import AzureOpenAIEmbeddings

embd = AzureOpenAIEmbeddings(
    azure_deployment="ada_gcal",
    openai_api_version="2024-02-01",
)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho"], embedding=embd
)
retriever = vectorstore.as_retriever()

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    # | ChatOpenAI(temperature=0)
    | ChatAnthropic(anthropic_api_key=anthropic_Api_Key)
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | ChatAnthropic() | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
