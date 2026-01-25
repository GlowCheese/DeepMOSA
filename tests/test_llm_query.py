import os

from pynguin import environ
from pynguin.configuration import config
from pynguin.llm.abstractmodel import AbstractLanguageModel


class LanguageModel(AbstractLanguageModel):
    def target_test_case(self, *args, **kwargs):
        pass


environ.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

config.llm.model = "deepseek-chat"
config.llm.base_url = "https://api.deepseek.com"

# config.llm.model = "deepseek/deepseek-chat"
# config.llm.base_url = "https://openrouter.ai/api/v1/chat/completions"


model = LanguageModel()

response = model.send_llm_request(
    [
        {
            "role": "user",
            "content": """what is 1+1?""",
        },
    ],
    stop=["\n```"],
    track_query_usage=False,
)


print(response)
