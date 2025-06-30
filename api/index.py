from backend_v2 import app
from vercel_fastapi import VercelFastAPI

vercel_app = VercelFastAPI(app)
app = vercel_app  # Vercel looks for 'app' 