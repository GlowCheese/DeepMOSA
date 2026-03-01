from mistralai import Mistral

from pynguin import environ

model = "devstral-2512"

client = Mistral(api_key=environ.OPENAI_API_KEY)

query = client.chat.complete(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "How far is the moon from earth?",
        },
    ],
)

response = query.choices[0]
print(response.finish_reason)
print(response.message.content)
print(query.usage)
