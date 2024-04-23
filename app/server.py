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

db = SQLDatabase.from_uri("sqlite:///Orders.db")
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


def llm_fallback(state):
    """
    Generate answer using the LLM w/o vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---LLM Fallback---")
    question = state["question"]
    generation = llm_chain.invoke({"question": question})
    return {"question": question, "generation": generation}

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    
    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue
    return {"documents": filtered_docs, "question": question}

def sql_database_search(state):
    """
    SQL database search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---NL TO SQL SEARCH (sql_database_search)---")
    question = state["question"]

    sql_based_answer = agent_executor.invoke({"input": question})
    return {"input": question,"documents": sql_based_answer["output"] }

### Edges ###

def route_question(state):
    """
    Route question to sql database search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    # route via tool calls
    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})
    
    # Fallback to LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("---ROUTE QUESTION TO LLM---")
        return "llm_fallback" 
    if len(source.additional_kwargs["tool_calls"]) == 0:
      raise "Router could not decide source"

    # Choose datasource
    print("choosing datasource")
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    print("choosing: ", datasource)
    if datasource == 'sql_database_search':
        print("---ROUTE QUESTION TO SQL DATABASE SEARCH---")
        return "sql_database_search"
    elif datasource == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        # print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, LLM FALLBACK---")
        return "llm_fallback"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
# def llm_fallback(state):
#     """
#     Generate answer using the LLM w/o vectorstore

#     Args:
#         state (dict): The current graph state

#     Returns:
#         state (dict): New key added to state, generation, that contains LLM generation
#     """
#     print("---LLM Fallback Activated---")
#     question = state["question"]
#     generation = test_query_chain.invoke({"question": question})
#     return {"question": question, "generation": generation}

# workflow.add_node("llm_fallback", llm_fallback) # llm

# workflow.set_entry_point("llm_fallback")
# workflow.add_edge("llm_fallback", END)

# workflow = StateGraph(GraphState)
# graph = workflow.compile()


# Build Graph
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("sql_database_search", sql_database_search) # database search
workflow.add_node("retrieve", retrieve) # rag retrieve
workflow.add_node("grade_documents", grade_documents) # grade documents
workflow.add_node("generate", generate) # rag generatae
# workflow.add_node("transform_query", transform_query) # transform_query
workflow.add_node("llm_fallback", llm_fallback) # llm


# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "sql_database_search": "sql_database_search",
        "vectorstore": "retrieve",
        "llm_fallback": "llm_fallback",

    },
)
workflow.add_edge("sql_database_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "sql_database_search": "sql_database_search",
        "generate": "generate",
        "llm_fallback": "llm_fallback"
    },
)
# workflow.add_edge("generate", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "llm_fallback",
        "useful": END,
        "not useful": "llm_fallback",
    },
)

workflow.add_edge("llm_fallback", END)
# TODO: Add edge that grades llm_fallback answer which doesn't mention AI Assistant, and only mentions rephrase question and transferred to live agent

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


# @app.get("/test")
# def retrieve_test():
#     generation = agent_executor.invoke(
#     "Who are the top 3 best selling artists?"
# )
#     return {
#         "message": generation,
#     }
    
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