"""Call a running server over HTTP with nothing but the standard library.

localdecision serve              # in another terminal
python examples/http_client.py
"""

import json
import os
import urllib.request

URL = os.environ.get("LOCALDECISION_URL", "http://127.0.0.1:8080") + "/v1/systemone"

with open(os.path.join(os.path.dirname(__file__), "ticket.json"), encoding="utf-8") as fh:
    payload = json.load(fh)

headers = {"Content-Type": "application/json"}
if key := os.environ.get("LOCALDECISION_API_KEY"):
    headers["Authorization"] = f"Bearer {key}"

request = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=headers)
with urllib.request.urlopen(request) as response:
    body = json.load(response)

for name, answer in body["answers"].items():
    print(name, json.dumps(answer))
print("timing:", body.get("timing"))
