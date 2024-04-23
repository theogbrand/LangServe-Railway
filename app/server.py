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

llm = ChatAnthropic()
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: "
        )
    ]
)

# Basic Test Query Chain
test_query_chain = prompt | llm | StrOutputParser()


### Build Documents Index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embd = AzureOpenAIEmbeddings(
    azure_deployment="ada_gcal",
    openai_api_version="2024-02-01",
)

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
# Question Router
question_router = route_prompt | structured_llm_router


### Retrieval Grader

# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

# LLM with function call 
llm = ChatCohere(model="command-r", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt 
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader


### Generation chain for synthesising full context

from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = hub.pull("rlm/rag-prompt")
print(prompt)
# LLM
llm = ChatCohere(model="command-r", temperature=0)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
rag_chain = prompt | llm | StrOutputParser()


### Hallucination Grader 

# Data model
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

# LLM with function call 
llm = ChatCohere(model="command-r", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# Prompt 
system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader = hallucination_prompt | structured_llm_grader
# hallucination_grader.invoke({"documents": docs, "generation": generation})


### Answer Grader 

# Data model
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

# LLM with function call 
llm = ChatCohere(model="command-r", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# Prompt 
system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader
# answer_grader.invoke({"question": question,"generation": generation})


### LLM fallback Chain

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage


# Preamble
preamble = """You are a fallback customer support assistant. You will be given a question you cannot answer based on internal knowledge. Suggest a way to rephrase their question in a way that relates to information regarding their most online order or product information. Reply the user politely that their request will be transferred to a live agent"""

# LLM
llm = ChatCohere(model_name="command-r", temperature=0).bind(preamble=preamble)

# Prompt
prompt = lambda x: ChatPromptTemplate.from_messages(
    [
        HumanMessage(
            f"Question: {x['question']} \nAnswer: "
        )
    ]
)

llm_chain = prompt | llm | StrOutputParser()


# SQL Agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import AzureChatOpenAI
llm = AzureChatOpenAI(
    deployment_name="pjf-dpo-turbo-35",
    api_version="2024-03-01-preview"
)
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
# print(db.dialect)
# print(db.get_usable_table_names())
agent_executor = create_sql_agent(llm, db=db, agent_type="openai-functions", verbose=True)


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
    documents : List[str]

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
    generation = agent_executor.invoke(
    "Who are the top 3 best selling artists?"
)
    return {
        "message": generation,
    }
    
    # generation = rag_chain.invoke({"context": docs, "question": "what is AG1"})
    # return {
    #     "message": generation,
    # }

    # question = "what is AG1?"
    # question = "what is agent memory?"
    # docs = retriever.get_relevant_documents(question)
    # doc_txt = docs[1].page_content
    # print(doc_txt)
    # response = retrieval_grader.invoke({"question": question, "document": doc_txt})
    # return {
    #     "message": response,
    # }
    
    # response = question_router.invoke({"question": "What ingredients does AG1 contain?"})
    # return {
    #     "message": response.response_metadata['tool_calls'],
    # }

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