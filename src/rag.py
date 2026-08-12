import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

load_dotenv()

FACTS_PATH = Path(__file__).parent / "student_facts.txt"

with open(FACTS_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

chunks = [p.strip() for p in raw_text.split("\n\n") if p.strip()]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_texts(chunks, embeddings)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


@tool
def answer_about_student(question: str) -> str:
    """Answer a question about the student (Mariana) using her personal facts.

    Use this tool whenever the user asks something about the student
    herself — her background, education, hobbies, or personal details.
    """
    relevant_chunks = vector_store.similarity_search(question, k=4)
    context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

    prompt = f"""Answer the question using only the context below. If the answer
    isn't explicitly stated in the context, do not calculate or infer facts
    (like age from a birth date) using assumptions that aren't given in the
    context. Instead, politely reply that you don't have access to that
    information.

    Context:
    {context}

    Question: {question}

    Answer:"""

    response = llm.invoke(prompt)
    return response.content


if __name__ == "__main__":
    query = "When was Mariana born?"
    print("Query: ", query)
    print("Answer:", answer_about_student.invoke(query))
    print("\n")
    query2 = "What are Mariana's hobbies?"
    print("Query: ", query2)
    print("Answer:", answer_about_student.invoke(query2))