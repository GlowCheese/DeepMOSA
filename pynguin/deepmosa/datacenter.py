from __future__ import annotations

import enum
import environ

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import Choice
from typing import List, Dict, Literal, Any, TYPE_CHECKING

from vendor.custom_logger import getLogger
from pynguin.runtimevar import RuntimeVariable
from pynguin.generic import GenericCallableAccessibleObject

from .api_errors import *

if TYPE_CHECKING:
    from pynguin.statistics.stats import StatisticsTracker


logger = getLogger(__name__)


"""                 -------------------------
                    | OPENAI DOCUMENTATIONS |
                    -------------------------

    Documentations on how prompt GPT for text generation with different
    criterions.

    - Quickstart to text-generation:
    https://platform.openai.com/docs/guides/text-generation?example=json

    - API reference:
    https://platform.openai.com/docs/api-reference/chat

    - Structured outputs:
    https://platform.openai.com/docs/guides/structured-outputs

    - Reasoning models:
    https://platform.openai.com/docs/guides/reasoning

                    -------------------------
                    |     MISCELLANEOUS     |
                    -------------------------

    Other documentations that might be useful.

    - Model endpoint compatibility:
    https://platform.openai.com/docs/models#model-endpoint-compatibility

    - Deprecations:
    https://platform.openai.com/docs/deprecations
"""


CONVERSATION = List[Dict[Literal['role', 'content'], Any]]


class DevPrompt(enum.Enum):
    NONE = None
    DEFAULT = """
Do NOT import pytest and unittest when writting test cases.
A good unit test should only contains variable assignments, assertions and function/method/constructor calls (i.e. without any custom class or function definition or control structure like `if`, `for`, `while`, `match`, `with`, ... statements).
All test cases should starts with: `def test_[test case's name]():`.
"""
    NO_CONTEXT = DEFAULT + """
Your response should only contain the test case itself without any additional text or information.
"""


class OpenAIDataCenter:
    """Make API requests and store cache for old conversations"""
    def __init__(self, temperature: float, statistics_tracker: StatisticsTracker | None = None):
        self.client = AsyncOpenAI()
        # map each gao to its initial conversation
        self._prompt: Dict[GenericCallableAccessibleObject, CONVERSATION] = {}
        self._input_tokens_cnt: int = 0
        self._output_tokens_cnt: int = 0

        self.temp = temperature
        self.stat = statistics_tracker


    def _record_query_usage(self, usage):
        self._input_tokens_cnt += int(usage.prompt_tokens)
        self._output_tokens_cnt += int(usage.completion_tokens)

        if self.stat is not None:
            self.stat.track_output_variable(
                RuntimeVariable.LLMInputTokens, self._input_tokens_cnt
            )
            self.stat.track_output_variable(
                RuntimeVariable.LLMOutputTokens, self._output_tokens_cnt
            )


    async def make_api_request(
        self,
        messages: CONVERSATION,
        dev_prompt: DevPrompt | None = DevPrompt.NONE
    ):
        """
        Make an API request to GPT and return its **text** response.

        You need to use `json.loads` to convert it to `dict` manually.
        """
        try:
            logger.info("Sending query using %s model", environ.OPENAI_CHAT_MODEL)
            sysrole = 'system' if environ.DEFAULT_MODEL == 'DEEPSEEK' else 'developer'

            if dev_prompt.value is not None:
                messages = [
                    {'role': sysrole, 'content': dev_prompt.value},
                    *messages
                ]
            query = await self.client.chat.completions.create(
                model=environ.OPENAI_CHAT_MODEL,
                messages=messages,
                temperature=self.temp,
                stream=False, stop="\n```"
            )
            response: Choice = query.choices[0]
            if response.finish_reason == "length":
                raise APILengthError()
            if response.message.refusal is not None:
                raise APIRefusalError(response.message.refusal)
            if response.finish_reason == "content_filter":
                raise APIContentFilterError()
            if response.finish_reason == "stop":
                self._record_query_usage(query.usage)
                resp = response.message.content
                return resp['test_case'] if isinstance(resp, dict) else resp

        except Exception as e:
            logger.exception("")
            raise APIUnknownError() from e


    def __getitem__(self, gao: GenericCallableAccessibleObject):
        """Return the GPT conversation for this gao.
        
        Returns:
            The conversation dictionary, or
            `None` if the conversation has not started.
        """
        return self._prompt.get(gao, None)


    def __setitem__(
        self,
        gao: GenericCallableAccessibleObject,
        initial_conversation: CONVERSATION
    ):
        """Set GPT initial conversation for this gao."""
        self._prompt[gao] = initial_conversation
