$ErrorActionPreference = "Stop"

env\Scripts\python.exe -m pip install fastapi uvicorn python-multipart pydantic
env\Scripts\python.exe -m uvicorn app.backend:app --host 0.0.0.0 --port 8000
