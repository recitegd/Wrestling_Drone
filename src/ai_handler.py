import requests
import os
import asyncio

ENDPOINT ="https://api.openai.com/v1/chat/completions"
MODEL = "gpt-5.4-mini"
CONTEXT = (
    "You are a concise folkstyle wrestling coach giving real-time mat advice. "
    "You will receive a spoken question plus separate vision entries for each wrestler. "
    "Use the wrestler labels exactly as given when comparing athletes. "
    "Base advice on posture, joint angles in degrees, and normalized joint positions. "
    "If the question targets one wrestler, answer for that wrestler. If unclear or ambiguous, default to both wrestlers. "
    "If vision data is missing or unclear, do not guess; include both wrestlers. "
    "Keep the response under 25 words and phrase it like a coach on the edge of the mat.\n"
)

headers = {
    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
    "Content-Type": "application/json"
}

class AiHandler:
    async def query(self, prompt):
        payload = {
            "model": MODEL,
            "messages": [
                    {"role": "system", "content": CONTEXT},
                    {"role": "user", "content": prompt}
                ]
        }

        try:
            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            print(CONTEXT + "\n" + prompt)
            print(data["choices"][0]["message"]["content"])
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            print("HTTP error:", e)
            print("Response text:", response.text)
