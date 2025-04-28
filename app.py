from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import MessageGraph
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableSequence
from rag import answer_from_rag
from rag_ticket import answer_from_ticket

import psycopg2
from datetime import datetime
import random
import json

# === Database Configuration ===
DB_PARAMS = {
    "dbname": "rag_db",
    "user": "nawafalhussain",
    "password": "postgres",
    "host": "localhost",
    "port": 5432,
}

# === LLMs ===
llm = OllamaLLM(model="llama3")
summarizer_llm = OllamaLLM(model="llama3")
technical_llm = OllamaLLM(model="llama3")

# === User Chat Memory ===
chat_memory: Dict[str, List] = {}

# === Ticket ID Generator ===
def generate_ticket_id() -> str:
    date_part = datetime.now().strftime("%Y%m%d")
    random_part = str(random.randint(100, 999))
    return f"TCK-{date_part}-{random_part}"

# === Save Escalation ===
def save_escalation_to_db(user_id: str, summary: str) -> str:
    ticket_id = generate_ticket_id()
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO helpdesk_escalations (user_id, message, ticket_id)
            VALUES (%s, %s, %s)
            """,
            (user_id, summary, ticket_id)
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"✅ Escalation saved: {ticket_id}")
        return ticket_id
    except Exception as e:
        print(f"❌ Failed to save escalation: {e}")
        return "TCK-ERROR"

# === Default Agent ===
def agent_node(messages: List):
    last_message = messages[-1].content
    response = llm.invoke(last_message)
    return AIMessage(content=response)

# === LangGraph ===
graph = MessageGraph()
graph.add_node("respond", agent_node)
graph.set_entry_point("respond")
chat_app = graph.compile()

# === Router Agent with JSON response ===
classification_prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an AI assistant that classifies user queries into categories:

    - "policy": HR-related topics (benefits, leave, salaries)
    - "ticket": IT-related issues (VPN, bugs, access)
    - "escalation": If the user requests to escalate the issue

    Respond in this exact JSON format only:
    {{
    "category": "policy"
    }}
    """),
    ("human", "{message}")
])
router_chain: Runnable = classification_prompt | llm

# === Summarizer ===
summarize_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Summarize the conversation below in 2-3 sentences."),
    ("human", "{history}")
])
summarize_chain: RunnableSequence = summarize_prompt | summarizer_llm

# === FastAPI App ===
app = FastAPI()

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.post("/chat")
async def chat_router(request: ChatRequest):
    user_id = request.user_id
    message = request.message

    history = chat_memory.get(user_id, [])
    history.append(HumanMessage(content=message))

    router_response = router_chain.invoke({"message": message})
    try:
        category = json.loads(router_response).get("category", "default")
    except Exception as e:
        print("❌ Router parsing failed:", e)
        category = "default"

    print(f"🧭 Router decided topic: {category}")

    if category == "policy":
        rag_result = answer_from_rag(message)
        response_content = rag_result["result"]
        sources = "\n".join([doc.page_content[:200] for doc in rag_result["source_documents"]])
        response_content += f"\n\n📎 Sources:\n{sources}"
        ai_message = AIMessage(content=response_content)

    elif category == "ticket":
        rag_result = answer_from_ticket(message)
        response_content = rag_result["result"]
        sources = "\n".join([doc.page_content[:200] for doc in rag_result["source_documents"]])
        response_content += f"\n\n📎 Sources:\n{sources}"
        ai_message = AIMessage(content=response_content)

    elif category == "escalation":
        full_history = "\n".join([msg.content for msg in history])
        summary = summarize_chain.invoke({"history": full_history})
        ticket_id = save_escalation_to_db(user_id, summary)

        response_content = (
            f"📌 Summary of your issue:\n{summary}\n\n"
            f"✅ Your issue has been escalated to the technical team.\n"
            f"🎫 Ticket ID: {ticket_id}\n"
            f"You can use this ID to follow up later."
        )
        ai_message = AIMessage(content=response_content)

    else:
        response = chat_app.invoke(history)
        ai_message = response[-1]

    history.append(ai_message)
    chat_memory[user_id] = history

    return {"response": ai_message.content}
