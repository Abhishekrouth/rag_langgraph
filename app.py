from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from typing import TypedDict
from langgraph.graph import StateGraph, END

load_dotenv()

app = Flask(__name__)

gemini_api_key = os.getenv("GEMINI_API_KEY")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=gemini_api_key
)

embedder = GoogleGenerativeAIEmbeddings(model="text-embedding-004",task_type="RETRIEVAL_DOCUMENT")

vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedder,
    text_key="text"
)

class AgentState(TypedDict):
    query: str
    issue_category: str
    purchase_type: str
    missing_info: bool
    context: list[str]
    response: str
    route: str

def classify_query(state: AgentState):
    prompt = f"""

Act as a classifier.

From the query, identify:
1. issue_category:
- defective_service
- delay_service
- refusal_to_honor_warranty
- overcharging_service
- unclear

2. purchase_type:
- online
- offline
- unknown

Query: {state["query"]}
"""
    result = llm.invoke(prompt).content.lower()

    if "defective" in result:
        state["issue_category"] = "defective_service"
    elif "delay" in result:
        state["issue_category"] = "delay_service"
    elif "refusal" in result:
        state["issue_category"] = "refusal_to_honor_warranty"
    elif "overcharging" in result:
        state["issue_category"] = "overcharging_service"
    else:
        state["issue_category"] = "unclear"

    if "online" in result:
        state["purchase_type"] = "online"
    elif "offline" in result:
        state["purchase_type"] = "offline"
    else:
        state["purchase_type"] = "unknown"

    if state["issue_category"] != "unclear":
        state["route"] = "retrieve_consumer_docs"
        state["missing_info"] = False
    else:
        state["route"] = "request_more_info"
        state["missing_info"] = True
    
    return state

def request_more_info(state: AgentState):
    state["response"] = (
        "To guide you correctly, always clarify:"
        "1. Was the purchase made online or offline?"
        "2. What issue are you facing (defective service, refund delay, service issue, overcharging issue)?"
    )
    return state

def retrieve_consumer_docs(state: AgentState):
    docs = vectorstore.similarity_search(state["query"], k=5)
    state["context"] = [doc.page_content for doc in docs]
    return state


def generate_consumer_response(state: AgentState):
    context_text = "\n".join(state["context"])
    prompt = f"""
You are a Consumer Rights Assistant for India.

Answer the user's questions strictly using the context below.
Base the answer on: context

If context is insufficient, clearly say more details are needed.

Context:
{context_text}

User Question:
{state["query"]}
"""

    state["response"] = llm.invoke(prompt).content
    return state

def route_after_classification(state: AgentState):
    return state.get("route", "request_more_info")


graph = StateGraph(AgentState)

graph.add_node("classify_query", classify_query)
graph.add_node("request_more_info", request_more_info)
graph.add_node("retrieve_consumer_docs", retrieve_consumer_docs)
graph.add_node("generate_consumer_response", generate_consumer_response)
graph.set_entry_point("classify_query")

graph.add_conditional_edges(
    "classify_query",
    route_after_classification,
    {
        "retrieve_consumer_docs": "retrieve_consumer_docs",
        "request_more_info": "request_more_info"
    }
)

graph.add_edge("request_more_info", END)
graph.add_edge("retrieve_consumer_docs", "generate_consumer_response")
graph.add_edge("generate_consumer_response", END)

consumer_rights_graph = graph.compile()
conversation_store = {}

@app.route('/', methods = ['GET'])
def app_check():
    return "App is running successfully"

@app.route('/index', methods = ['POST'])
def index_docs():

    loader = PyPDFLoader('ConsumerRights.pdf')
    documents = loader.load()

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = splitter.split_documents(documents)
    
    vectors = []
    for i, doc in enumerate(chunks):
        vec = embedder.embed_documents([doc.page_content])[0]
        vectors.append((str(i),vec,{"text": doc.page_content}))
    index.upsert(vectors)

    print("Chunks are:", chunks[:2])

    return jsonify({
            "status": "PDF indexed",
            "Number of chunks": len(chunks)
            })

@app.route('/ask_query', methods = ['POST'])
def ask():
    
    query = request.json.get("query")
    query_vec = embedder.embed_query(query)
    
    results = index.query(
        vector=query_vec,
        top_k=5,
        include_metadata=True
    )
    
    context = " ".join(
        match["metadata"]["text"] for match in results["matches"]
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a Consumer rights assistant of India. Reply in a customer representative tone.
    Answer only using the below context.
    If user ask any irrelevant question, reply with "Please ask questions about Consumer Rights"

    Context:
    {context}

    Question:
    {query}
    """)

    parser = StrOutputParser()
    chain = prompt | llm | parser

    n = {"context": context,"query": query}
    output = chain.invoke(n)

    return jsonify({
        "answer": output
    })


@app.route("/ask_query_langgraph", methods=["POST"])
def langgraph():

    data = request.get_json()
    session_id = data.get("session_id")
    query = data.get("query")

    if not session_id or not query:
        return jsonify({"error": "session_id and query required"}), 400

    if session_id not in conversation_store:
        conversation_store[session_id] = {
            "query": "",
            "context": [],
            "response": None,
            "route": None
        }

    state = conversation_store[session_id]
    state["query"] = query

    result = consumer_rights_graph.invoke(state)
    conversation_store[session_id] = result

    return jsonify({
        "answer": result["response"]
    })

if __name__ == "__main__":
    app.run(debug=True)
