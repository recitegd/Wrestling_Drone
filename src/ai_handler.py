import requests
import os
import asyncio

ENDPOINT ="https://api.openai.com/v1/chat/completions"
MODEL = "gpt-5.4-mini"
CONTEXT = "You are a master folkstyle wrestling coach and you need to give instructions and tips to positions you see based on joint angles in degrees given to you and questions that you may be asked. Keep your response very short, within a few phrases. Also, please reply quickly.\n"

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