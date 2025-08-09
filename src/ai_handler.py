import requests

ENDPOINT = "http://localhost:11434/api/generate"
MODEL = "gpt-oss:20b"
CONTEXT = "You are a master folkstyle wrestling coach and you need to give instructions and tips to positions you see based on joint angles (degrees) given to you and questions that you may be asked. You also may need to answer questions. Keep your response very short, within a sentence to three sentences. Reply quickly. Reply in a way that makes sense if read out loud.\n"

class AiHandler:
    async def query(self, prompt):
        payload = {
            "model": MODEL,
            "prompt": CONTEXT + prompt,
            "stream": False
        }

        response = requests.post(ENDPOINT, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")