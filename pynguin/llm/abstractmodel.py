import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion

import pynguin.utils.statistics.stats as stat
from libs.custom_logger import getLogger
from pynguin import environ
from pynguin.configuration import config
from pynguin.utils.deepseek import tokenizer
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

from .api_errors import APIContentFilterError, APIRefusalError

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

    def __log_messages_stats(self, messages: Messages):
        _logger.info("Sending query to model: %s", config.llm.model)
        num_chars = sum(len(m["content"]) for m in messages)
        num_tokens = sum(len(tokenizer.encode(m["content"])) for m in messages)
        _logger.info("Query size: %s characters (~%s tokens)", num_chars, num_tokens)

    def __handle_llm_query(self, query: ChatCompletion, query_at: float, track_query_usage: bool):
        response = query.choices[0]
        if response.finish_reason == "content_filter":
            raise APIContentFilterError()
        if response.message.refusal is not None:
            raise APIRefusalError(response.message.refusal)

        if response.finish_reason == "length":
            _logger.warning("LLM output truncated due to token limit")
        else:
            assert response.finish_reason == "stop"

        if track_query_usage:
            assert query.usage is not None

            self._num_llm_calls += 1
            self._time_calling_llm += time.time() - query_at
            self._input_tokens_cnt += query.usage.prompt_tokens
            self._output_tokens_cnt += query.usage.completion_tokens

            _logger.info("Output size: %s tokens", query.usage.completion_tokens)

            stat.track_output_variable(RuntimeVariable.LLMCalls, self._num_llm_calls)
            stat.track_output_variable(RuntimeVariable.LLMQueryTime, self._time_calling_llm)
            stat.track_output_variable(RuntimeVariable.LLMInputTokens, self._input_tokens_cnt)
            stat.track_output_variable(RuntimeVariable.LLMOutputTokens, self._output_tokens_cnt)

        assert response.message.content is not None
        return response.message.content

    def send_llm_request(
        self, messages: Messages, *, stop: str | List[str], track_query_usage=True
    ):
        client = OpenAI(api_key=environ.OPENAI_API_KEY, base_url=config.llm.base_url)
        query_at = time.time()
        self.__log_messages_stats(messages)
        query = client.chat.completions.create(
            messages=messages,  # type: ignore
            model=config.llm.model,
            temperature=config.llm.temperature,
            stream=False,
            stop=stop,
            max_tokens=config.llm.max_tokens,
        )
        return self.__handle_llm_query(query, query_at, track_query_usage)

    async def send_llm_request_async(
        self, messages: Messages, *, stop: str | List[str], track_query_usage=True
    ):
        client = AsyncOpenAI(api_key=environ.OPENAI_API_KEY, base_url=config.llm.base_url)
        query_at = time.time()
        self.__log_messages_stats(messages)
        query = await client.chat.completions.create(
            messages=messages,  # type: ignore
            model=config.llm.model,
            temperature=config.llm.temperature,
            stream=False,
            stop=stop,
            max_tokens=config.llm.max_tokens,
        )
        return self.__handle_llm_query(query, query_at, track_query_usage)

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

    def _log_query_data(self, file_name: str, data: str, header: str):
        report_dir = config.statistics_output.report_dir
        file_path = Path(report_dir) / "llm" / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "a+", encoding="UTF-8") as file:
            if self._num_llm_calls == 1:
                file.write(
                    "####################################################################\n"
                    f"# {f'TEST GENERATION BEGINS ({config.algorithm.name} + {config.llm.model} t={config.llm.temperature})':^64} #\n"
                    "####################################################################\n\n\n"
                )

            file.write(
                f"# {header} at query #{self._num_llm_calls}\n"
                "#--------------------------\n\n"
                f"{data}\n\n\n"
            )
