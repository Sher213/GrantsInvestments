from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import pandas as pd

# Load env variables
load_dotenv()

DB_URL = os.getenv("DATABASE_URL")
TABLE_NAME = "grants"
engine = create_engine(DB_URL)

app = FastAPI()

# Allow Streamlit frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/grants")
def get_grants():
    with engine.connect() as conn:
        df = pd.read_sql_table(TABLE_NAME, conn)
        return df.to_dict(orient="records")