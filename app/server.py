from fastapi import FastAPI
from langserve import add_routes
# from pirate_speak.chain import chain as pirate_speak_chain
# add_routes(app, pirate_speak_chain, path="/pirate-speak")


import os
from dotenv import load_dotenv
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate

load_dotenv()

anthropic_Api_Key: str = os.environ["ANTHROPIC_API_KEY"]


### LLM fallback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage


# LLM
llm = ChatAnthropic()

# Prompt
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: "
        )
    ]
)


### Build Index

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# from langchain_openai import OpenAIEmbeddings
### from langchain_cohere import CohereEmbeddings

# Set embeddings
embd = AzureOpenAIEmbeddings(
    azure_deployment="ada_gcal",
    openai_api_version="2024-02-01",
)

# Docs to index
urls = [
    # "https://lilianweng.github.io/posts/2023-06-23-agent/",
    # "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    # "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
    "https://drinkag1.com/foundational-nutrition-education",
    "https://drinkag1.com/about-ag1/quality-standards/ctr",
    "https://drinkag1.com/about-ag1/ingredients/ctr"

]

# Load
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=500, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# # Add to vectorstore
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag_adapt",
    embedding=embd,
)
retriever = vectorstore.as_retriever()



### Router
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import ChatCohere

# Data model
class sql_database_search(BaseModel):
    """
    A sql database containing the ORDERS table, which stores data about the customer's order information including order shipment details, last payment made and subscription status.
    """
    query: str = Field(description="The query to use when fetching information from the sql database.")

class vectorstore(BaseModel):
    """
    A vectorstore containing documents related to product information, Athletic Greens, also known as AG1. Use the vectorstore for questions on details relating to what the product is, how the product was manufactured and the story behind the product.
    """
    query: str = Field(description="The query to use when searching the vectorstore.")

# Preamble
preamble = """You are an expert at routing a user question to a vectorstore or sql_database.
The vectorstore contains documents related to product details. Use the vectorstore for questions relating to information about the product, Athletic Greens. Otherwise, use the sql_database to answer questions relating to customer order information, such as order shipment details, last payment made and subscription status."""

# LLM with tool use and preamble
llm = ChatCohere(model="command-r", temperature=0)
structured_llm_router = llm.bind_tools(tools=[sql_database_search, vectorstore], preamble=preamble)

# Prompt
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router
# response = question_router.invoke({"question": "Where is my order?"})
# print(response.response_metadata['tool_calls'])
# response = question_router.invoke({"question": "What ingredients does AG1 contain?"})
# print(response.response_metadata['tool_calls'])
# response = question_router.invoke({"question": "Hi how are you?"})
# print('tool_calls' in response.response_metadata)


# Chain
test_query_chain = prompt | llm | StrOutputParser()


from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents 
    """
    question : str
    generation : str
    # documents : List[str]

workflow = StateGraph(GraphState)
def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---LLM Fallback Activated---")
    question = state["question"]
    generation = test_query_chain.invoke({"question": question})
    return {"question": question, "generation": generation}

workflow.add_node("llm_fallback", llm_fallback) # llm

workflow.set_entry_point("llm_fallback")
workflow.add_edge("llm_fallback", END)

graph = workflow.compile()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)

@app.get("/")
def root():
    return {
        "message": "serving healthy",
    }


@app.get("/test")
def retrieve_test():
    response = question_router.invoke({"question": "What ingredients does AG1 contain?"})
    return {
        "message": response.response_metadata['tool_calls'],
    }

    # question = "agent memory?"
    # docs = retriever.get_relevant_documents(question)
    # return {
    #     "message": docs[1].page_content,
    # }


@app.get("/ping")
def health_check():
    return {"status": "PONG"}

from langchain_core.runnables import chain


@chain
def custom_chain(json_input: dict):
  inputs = {"question": json_input["question"]}
  return graph.invoke(inputs)

add_routes(
    app,
    custom_chain,
    # convo_chain,
    path="/chat"
)

# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
# add_routes(app, convo_chain, enable_feedback_endpoint=True)

# from fastapi.middleware.cors import CORSMiddleware

# # Set all CORS enabled origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
#     expose_headers=["*"],
# )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)