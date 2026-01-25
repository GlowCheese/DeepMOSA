import json

import requests

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/json",
}

payload = {
    "model": "mistralai/devstral-2512:free",
    # "model": "anthropic/claude-haiku-4.5",
    # "model": "deepseek/deepseek-chat",
    "messages": [
        {
            "role": "developer",
            "content": (
                "Write unit test for the given code object without any additional text or information.\n"
                "Do not write any import statement (assuming everything is correctly imported)."
            ),
        },
        {
            "role": "user",
            "content": """
gcd.py:
```
def gcd(a, b):
    if a == b:
        return a
    if a < b:
        return gcd(a, b - a)
    else:
        return gcd(a - b, b)
```

Write a unit test with pytest for the function `gcd` with the following signature: `def test_gcd()`.
""",
        },
    ],
}

resp = requests.post(url, headers=headers, data=json.dumps(payload))
# print(resp.json())
print(resp.json()["choices"][0]["message"]["content"])
