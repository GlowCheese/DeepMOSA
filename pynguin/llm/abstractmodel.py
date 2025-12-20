import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

import pynguin.utils.statistics.stats as stat
from pynguin import environ
from pynguin.configuration import config
from pynguin.utils.custom_logger import getLogger
from pynguin.utils.deepseek import tokenizer
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

from .api_errors import APIContentFilterError, APILengthError, APIRefusalError

_logger = getLogger(__name__)


Messages = List[Dict[Literal["role", "content", "prefix"], Any]]


class AbstractLanguageModel(ABC):
    """An interface for an OpenAI language model to generate/mutate tests as natural language."""

    def __init__(self):
        self.test_src: str
        self._max_query_len: int = 4000
        self._token_len_cache = {}

        # statistics
        self._num_llm_calls = 0
        self._time_calling_llm = 0.0
        self._input_tokens_cnt = 0
        self._output_tokens_cnt = 0

    def __handle_llm_query(self, query: ChatCompletion, query_at: float):
        response = query.choices[0]
        if response.finish_reason == "length":
            raise APILengthError()
        if response.message.refusal is not None:
            raise APIRefusalError(response.message.refusal)
        if response.finish_reason == "content_filter":
            raise APIContentFilterError()

        assert response.finish_reason == "stop"
        assert query.usage is not None

        self._num_llm_calls += 1
        self._time_calling_llm += time.time() - query_at
        self._input_tokens_cnt += int(query.usage.prompt_tokens)
        self._output_tokens_cnt += int(query.usage.completion_tokens)

        stat.track_output_variable(RuntimeVariable.LLMCalls, self._num_llm_calls)
        stat.track_output_variable(RuntimeVariable.LLMQueryTime, self._time_calling_llm)
        stat.track_output_variable(RuntimeVariable.LLMInputTokens, self._input_tokens_cnt)
        stat.track_output_variable(RuntimeVariable.LLMOutputTokens, self._output_tokens_cnt)

        assert response.message.content is not None
        return response.message.content

    def send_llm_request(self, messages: Messages, *, stop: str | List[str]):
        client = OpenAI(api_key=environ.OPENAI_API_KEY, base_url=config.llm.base_url)
        query_at = time.time()
        query = client.chat.completions.create(
            messages=messages,  # type: ignore
            model=config.llm.model,
            temperature=config.llm.temperature,
            stream=False,
            stop=stop,
            max_tokens=config.llm.max_tokens,
        )
        return self.__handle_llm_query(query, query_at)

    async def send_llm_request_async(self, messages: Messages, *, stop: str | List[str]):
        client = AsyncOpenAI(api_key=environ.OPENAI_API_KEY, base_url=config.llm.base_url)
        query_at = time.time()
        _logger.info(
            "Sending query to model: %s (temp %s)", config.llm.model, config.llm.temperature
        )
        query = await client.chat.completions.create(
            messages=messages,  # type: ignore
            model=config.llm.model,
            temperature=config.llm.temperature,
            stream=False,
            stop=stop,
            max_tokens=config.llm.max_tokens,
        )
        return self.__handle_llm_query(query, query_at)

    @abstractmethod
    def target_test_case(self, *args, **kwargs):
        pass

    def _get_num_tokens_at_line(self, line_num: int) -> int:
        """Get the approximate number of tokens for the source file at line_num.

        Args:
            line_num: the line number to get the number of tokens for

        Returns:
            the approximate number of tokens
        """
        if len(self._token_len_cache) == 0:
            self._token_len_cache = {
                i + 1: len(tokenizer.encode(line))
                for i, line in enumerate(self.test_src.split("\n"))
            }
        return self._token_len_cache[line_num]

    def _log_prompt_used_and_response(
        self, prompt: str, raw_generated_test: str, generated_test_after_fixup: str
    ):
        """Log conversation and generated unit test for debugging purpose."""

        raise NotImplementedError()

        # now = datetime.now()

        # with open(
        #     os.path.join(config.statistics_output.report_dir, "gpt_raw_generated.py"),
        #     "a+",
        #     encoding="UTF-8",
        # ) as log_file:
        #     log_file.write(f"\n\n# ({config.module_name}) Generated at {now}\n")
        #     log_file.write(raw_generated_test)

        # with open(
        #     os.path.join(config.statistics_output.report_dir, "gpt_generated_after_fixup.py"),
        #     "a+",
        #     encoding="UTF-8",
        # ) as log_file:
        #     log_file.write(f"\n\n# ({config.module_name}) Generated at {now}\n")
        #     log_file.write(generated_test_after_fixup)

        # with open(
        #     os.path.join(config.statistics_output.report_dir, "gpt_prompts.py"),
        #     "a+",
        #     encoding="UTF-8",
        # ) as log_file:
        #     log_file.write(f"\n\n# ({config.module_name}) prompt sent at {now}\n")
        #     log_file.write(prompt)
