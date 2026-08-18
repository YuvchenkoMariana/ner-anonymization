from fastapi import FastAPI
from pydantic import BaseModel

from rag import answer_about_student

app = FastAPI(title="RAG Service")


class QuestionRequest(BaseModel):
    question: str


@app.post("/answer")
def answer(request: QuestionRequest):
    return {"result": answer_about_student.invoke(request.question)}
