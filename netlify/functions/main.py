from mangum import Mangum
from main import app  # Import FastAPI app from main.py

handler = Mangum(app) 