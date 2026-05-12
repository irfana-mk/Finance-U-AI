from .llm import ask_gemini
from .ollama import ask_ollama
from .rag import retrieve_context

SYSTEM_PROMPT = """You are a personal finance assistant.
If user describes a transaction (like "I spent 500 on food" or "Add income 2000"), return JSON ONLY.
Example transaction JSON (must be valid JSON):
{
"action": "add_transaction",
"type": "expense",
"amount": 500,
"category": "food"
}
If user asks a question -> respond normally.
Never mix JSON and text."""

def build_prompt(user, message):
    # Load user context
    expenses = user.expenses.all().order_by('-created_at')[:5]
    user_context = "Your Recent Expenses:\n"
    if expenses.exists():
        for exp in expenses:
            user_context += f"- {exp.category}: {exp.amount}\n"
    else:
        user_context += "No expenses yet.\n"
        
    rag_context = retrieve_context(message)
    
    prompt = f"""{SYSTEM_PROMPT}

[USER FINANCIAL CONTEXT]
{user_context}

[KNOWLEDGE BASE CONTEXT]
{rag_context}

[USER MESSAGE]
{message}
"""
    return prompt

def ask_ai(user, message: str) -> str:
    prompt = build_prompt(user, message)
    
    try:
        # 1. Try Gemini
        return ask_gemini(prompt)
    except Exception as e:
        print(f"Gemini failed: {e}. Falling back to Ollama.")
        try:
            # 2. Try Ollama Fallback
            return ask_ollama(prompt, model="llama3")
        except Exception as e2:
            print(f"Ollama failed: {e2}.")
            # 3. Complete Fallback
            return "Sorry, my AI systems are currently offline. However, based on my local database: " + retrieve_context(message)
