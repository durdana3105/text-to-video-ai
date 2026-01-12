from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Text to Video API is running"}
