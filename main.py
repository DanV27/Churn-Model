from fastapi import FastAPI

# Initialize the application instance
app = FastAPI()

# Define a path operation decorator for HTTP GET requests
@app.get("/")
def read_root():
    return {"status": "success", "message": "Hello World"}

# Define an endpoint with a path parameter and query parameter
@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "query_parameter": q}
