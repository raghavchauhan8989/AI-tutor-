import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
app = FastAPI()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
sessions = {}

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.get("/")
def read_root():
    return {"message": "AI Tutor is running!"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    if request.session_id not in sessions:
        sessions[request.session_id] = []
    
    docs = retriever.invoke(request.question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    history = sessions[request.session_id]
    
    messages = [
        SystemMessage(content=f"You are an expert AI tutor. Use ONLY the following context to answer the student's question. If the answer is not in the context, say 'I don't have that information in the course material.'\n\nContext:\n{context}")
    ]
    
    for h in history:
        messages.append(HumanMessage(content=h["question"]))
        messages.append(SystemMessage(content=h["answer"]))
    
    messages.append(HumanMessage(content=request.question))
    
    response = llm.invoke(messages)
    answer = response.content
    
    history.append({"question": request.question, "answer": answer})
    sessions[request.session_id] = history[-10:]
    
    return {"answer": answer, "session_id": request.session_id}

@app.get("/health")
def health():
    return {"status": "ok"}


class QuizRequest(BaseModel):
    topic: str
    session_id: str = "default"

@app.post("/quiz")
def generate_quiz(request: QuizRequest):
    docs = retriever.invoke(request.topic)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    prompt = f"""Based on the following context, generate exactly 5 multiple choice questions about "{request.topic}".

Context:
{context}

Return ONLY a JSON array like this:
[
  {{
    "question": "Question text here?",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "answer": "A) option1"
  }}
]
Return only the JSON array, nothing else."""

    messages = [HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    
    import json
    try:
        clean = response.content.strip().replace("```json", "").replace("```", "")
        questions = json.loads(clean)
    except:
        questions = response.content
    
    return {"topic": request.topic, "questions": questions}