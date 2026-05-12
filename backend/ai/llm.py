import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return the response."""
    if not api_key:
        raise Exception("GEMINI_API_KEY is not set.")
    
    # We use gemini-1.5-flash for general fast interaction
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text
