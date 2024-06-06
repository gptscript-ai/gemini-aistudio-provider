import json
import os
import subprocess
import sys

if 'GEMINI_API_KEY' in os.environ:
    print("Taking API Key from GEMINI_API_KEY environment variable.", file=sys.stderr)
    output = {
        "env": {
            "GEMINI_API_KEY": os.environ['GEMINI_API_KEY'],
        }
    }
    print(json.dumps(output))
    sys.exit(0)

tool_input = {
    "message": "Please enter your Google AI Studio API key.",
    "fields": "key",
}
command = ["gptscript", "--quiet=true", "--disable-cache", "sys.prompt", json.dumps(tool_input)]
result = subprocess.run(command, stdin=None, stdout=subprocess.PIPE, text=True)
if result.returncode != 0:
    print("Failed to run sys.prompt.", file=sys.stderr)
    sys.exit(1)

try:
    resp = json.loads(result.stdout.strip())
    key = resp["key"]

except json.JSONDecodeError:
    print("Failed to decode JSON.", file=sys.stderr)
    sys.exit(1)

output = {
    "env": {
        "GEMINI_API_KEY": key,
    }
}

print(json.dumps(output))
