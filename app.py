from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from typing import TypedDict, List
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
    context: List[str]
    response: str
    route: str

def classify_query(state: AgentState):
    text = state["query"].lower()

    if not state["issue_category"]:
        if any(k in text for k in ["defective", "broken", "faulty", "not working"]):
            state["issue_category"] = "defective_service"
            
        elif any(k in text for k in ["delay", "late", "not delivered", "refund delay"]):
            state["issue_category"] = "delay_service"

        elif any(k in text for k in ["warranty", "refused", "denied"]):
            state["issue_category"] = "refusal_to_honor_warranty"

        elif any(k in text for k in ["overcharge", "overcharged", "extra charge", "hidden fee"]):
            state["issue_category"] = "overcharging_service"

    if not state["issue_category"]:
        state["missing_info"] = True
        state["route"] = "request_more_info"

    else:
        state["missing_info"] = False
        state["route"] = "retrieve_docs"

    return state

def request_more_info(state: AgentState):
    state["response"] = (
        "I need a bit more information to help you."
        "Can you state the issue category?"
    )
    return state

def retrieve_docs(state: AgentState):
    docs = vectorstore.similarity_search(state["query"], k=5)
    state["context"] = [doc.page_content for doc in docs]
    return state

def generate_answer(state: AgentState):
    context_text = "\n".join(state["context"])
    prompt = f"""
You are a Consumer Rights Assistant for India. Answer strictly using the context below.
If the context doesn't specifies the issue category, ask for the issue category.

Context:
{context_text}

Question:
{state["query"]}

""" 
    state["response"] = llm.invoke(prompt).content
    return state

def route_logic(state: AgentState):
    return state["route"]

graph = StateGraph(AgentState)

graph.add_node("classify_query", classify_query)
graph.add_node("request_more_info", request_more_info)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("generate_answer", generate_answer)

graph.set_entry_point("classify_query")

graph.add_conditional_edges(
    "classify_query",
    route_logic,
    {
        "request_more_info": "request_more_info",
        "retrieve_docs": "retrieve_docs"
    }
)

graph.add_edge("request_more_info", END)
graph.add_edge("retrieve_docs", "generate_answer")
graph.add_edge("generate_answer", END)

consumer_graph = graph.compile()
conversation_store = {}
    
@app.route("/", methods=["GET"])
def health():
    return "App is working"

@app.route("/index", methods=["POST"])
def index_docs():

    loader = DirectoryLoader(
        "Doc/", 
        glob="**/*.pdf", 
        loader_cls=PyPDFLoader, 
        show_progress=True)
    documents = loader.load()
    
    chunk_size=300
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    vectors = []
    for i, doc in enumerate(chunks):
        vec = embedder.embed_documents([doc.page_content])[0]
        vectors.append((str(i), vec, {"text": doc.page_content}))

    index.upsert(vectors)
    chunks_ex = chunks[:3]
    print("Chunks examples are:", chunks_ex)
    
    return jsonify(
        {
            "status": "Documents are indexed successfully", 
            "chunks": len(chunks)
            }
        )

@app.route('/ask_query', methods = ['POST'])
def ask():
    
    query = request.json.get("query")
    query_vec = embedder.embed_query(query)
    
    results = index.query(
        vector=query_vec,
        top_k=3,
        include_metadata=True
    )
    
    context = " ".join(match["metadata"]["text"] for match in results["matches"])

    prompt = ChatPromptTemplate.from_template("""
    You are a Consumer rights assistant of India. Reply in a customer representative tone.
    Answer only using the below context. If user ask any irrelevant question, reply with:
    "Please ask questions about Consumer Rights"

    Context:
    {context}

    Question:
    {query}   """
    )

    parser = StrOutputParser()
    chain = prompt | llm | parser

    n = {"context": context,"query": query}
    output = chain.invoke(n)

    return jsonify({
        "answer": output,
        "Application":"Normal RAG"
    })

@app.route("/ask_query_langgraph", methods=["POST"])
def ask_langgraph():
    
    data = request.get_json()
    session_id = data.get("session_id")
    query = data.get("query")

    if not session_id or not query:
        return jsonify(
            {
                "error": "Please enter both session_id and query"
             }
        ), 400

    if session_id not in conversation_store:
        conversation_store[session_id] = {
            "query": "",
            "issue_category": "",
            "purchase_type": "",
            "missing_info": False,
            "context": [],
            "response": "",
            "route": ""
        }

    state = conversation_store[session_id]
    state["query"] = (state["query"] + " " + query).strip()

    result = consumer_graph.invoke(state)
    conversation_store[session_id] = result
    
    return jsonify({
        "answer": result["response"],
        "missing_info": result["missing_info"],
        "session_name": session_id,
        "Application": "RAG using LangGraph"
    })

if __name__ == "__main__":
    app.run(debug=True)