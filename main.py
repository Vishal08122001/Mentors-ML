from fastapi import FastAPI
from pydantic import BaseModel
from model.model import main
from pandas import DataFrame
from fastapi.responses import Response
import json

app = FastAPI()

class TextIn(BaseModel):
    employeeCode: str

@app.get('/')
def home():
    return {"Mentors_Recommendation": "OK", "Version": "0.1.0"}

@app.post("/recommend")
async def recommend(payload: TextIn) -> Response:
    mentors = main(payload.employeeCode)
    # Convert DataFrame to JSON string with indentation for readability
    json_data = mentors.to_json(indent=4, orient="records")
    return Response(content=json_data, media_type="application/json")
