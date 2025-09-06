import os
import asyncio
from pynguin.globl import Globl
from pynguin.deepmosa import OpenAIDataCenter
from pynguin.deepmosa.datacenter import DevPrompt


async def test_make_api_request():
    conversation = [
        {
            "role": "user",
            "content": "Who is GlowCheese?"
        }
    ]
    datacenter = OpenAIDataCenter(0.8)
    response = await datacenter.make_api_request(
        conversation,
        DevPrompt.NONE
    )
    with open(".trash/tmp.txt", "w", encoding='utf-8') as file:
        file.write(response)

    print(response)