from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="FastAPI Backend", version="1.0.0")

# Request body model for validation
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float

# Root GET route
@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!", "status": "success"}
