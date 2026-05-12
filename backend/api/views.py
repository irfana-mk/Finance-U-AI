import json
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from ai.logic import ask_ai
from finance.models import Expense

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def chat_view(request):
    message = request.data.get('message', '')
    if not message:
        return Response({"error": "Message is required"}, status=status.HTTP_400_BAD_REQUEST)

    try:
        reply = ask_ai(request.user, message)
        
        # Strip codeblock wrappers if Gemini returns markdown JSON
        clean_reply = reply.strip()
        if clean_reply.startswith('```json'):
            clean_reply = clean_reply[7:-3].strip()
        elif clean_reply.startswith('```'):
            clean_reply = clean_reply[3:-3].strip()

        try:
            # Check if it's a valid JSON transaction
            data = json.loads(clean_reply)
            if data.get("action") == "add_transaction" and data.get("type") == "expense":
                Expense.objects.create(
                    user=request.user,
                    category=data.get("category", "General"),
                    amount=data.get("amount", 0.0)
                )
                return Response({
                    "reply": f"Recorded expense of {data.get('amount')} for {data.get('category')}."
                })
        except json.JSONDecodeError:
            # Not JSON, normal reply
            pass

        return Response({"reply": reply})
    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
