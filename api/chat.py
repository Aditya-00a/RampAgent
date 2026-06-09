"""Vercel serverless function — AI chat endpoint powered by Cerebras."""

import json
import os
from http.server import BaseHTTPRequestHandler
from openai import OpenAI


CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")
CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_MODEL = "llama3.1-8b"

# Load pre-computed results for context
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "data", "results.json")
try:
    with open(DATA_PATH) as f:
        RESULTS = json.load(f)
    SUMMARY_CONTEXT = json.dumps(RESULTS["summary"], indent=2)
    FLAGGED_BRIEF = json.dumps(RESULTS["flagged_transactions"][:15], indent=2)
    KRI_CONTEXT = json.dumps(RESULTS.get("kris", []), indent=2)
    REGISTER_CONTEXT = json.dumps(RESULTS.get("risk_register", []), indent=2)
except Exception:
    SUMMARY_CONTEXT = "{}"
    FLAGGED_BRIEF = "[]"
    KRI_CONTEXT = "[]"
    REGISTER_CONTEXT = "[]"


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length else {}
        question = body.get("question", "")

        if not question:
            self._respond(400, {"error": "Missing 'question' field"})
            return

        if not CEREBRAS_API_KEY:
            self._respond(500, {"error": "CEREBRAS_API_KEY not configured"})
            return

        try:
            client = OpenAI(base_url=CEREBRAS_BASE_URL, api_key=CEREBRAS_API_KEY)
            system_prompt = f"""You are an AI risk analyst for an enterprise risk & controls monitoring platform covering corporate spend. Answer questions using the data below. Speak the language of a risk / internal audit team: exposure, control exceptions, risk rating, likelihood and impact. Be specific — cite dollar amounts, names, percentages, risk IDs and control IDs.

SUMMARY (incl. exposure & expected loss):
{SUMMARY_CONTEXT}

KEY RISK INDICATORS:
{KRI_CONTEXT}

RISK REGISTER:
{REGISTER_CONTEXT}

TOP CONTROL EXCEPTIONS / RISK EVENTS:
{FLAGGED_BRIEF}"""

            response = client.chat.completions.create(
                model=CEREBRAS_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content.strip()
            self._respond(200, {"answer": answer})
        except Exception as e:
            self._respond(500, {"error": str(e)})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _respond(self, status, data):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
