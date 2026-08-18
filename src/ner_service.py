from fastapi import FastAPI
from pydantic import BaseModel

from anonymize import anonymize_text
from anonymize_uk import anonymize_text_uk

app = FastAPI(title="NER Service")


class AnonymizeRequest(BaseModel):
    text: str


@app.post("/anonymize/en")
def anonymize_en(request: AnonymizeRequest):
    return {"result": anonymize_text.invoke(request.text)}


@app.post("/anonymize/uk")
def anonymize_uk(request: AnonymizeRequest):
    return {"result": anonymize_text_uk.invoke(request.text)}