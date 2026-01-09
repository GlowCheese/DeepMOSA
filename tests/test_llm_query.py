# from pynguin.llm.abstractmodel import AbstractLanguageModel


# class LanguageModel(AbstractLanguageModel):
#     def target_test_case(self, *args, **kwargs):
#         pass


# model = LanguageModel()


import os

from litellm import completion

os.environ["DEEPSEEK_API_KEY"] = "sk-4723574ff4c242a8bd6a84eabde28ba3"

response = completion(
    model="deepseek/deepseek-chat",
    messages=[
        {"role": "user", "content": "Who won the world cup in 2022?"},
        {"role": "assistant", "content": "Argentina", "prefix": True},
    ],
)
print(response.choices[0].message.content)
