from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, cast

from omegaconf import OmegaConf

import pynguin.utils.statistics.stats as stat
from libs.custom_logger import getLogger
from pynguin.configuration import config
from pynguin.utils.deepseek import tokenizer
from pynguin.utils.statistics.runtimevariable import RuntimeVariable

if TYPE_CHECKING:
    from langchain_core.messages import UsageMetadata
    from langchain_core.prompts import ChatPromptTemplate


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

        # LLM client
        cfg_path = Path(__file__).parent / "endpoints" / f"{config.llm.llm_config_id}.yml"
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.set_struct(cfg, True)
        llm_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.llm, resolve=True))

        match cfg.llm_provider:  # pyright: ignore[reportAttributeAccessIssue]
            case "ollama":
                from langchain_ollama import ChatOllama

                self.llm = ChatOllama(
                    **llm_cfg,
                    stop=["\n````"],
                    temperature=config.llm.temperature,
                )
            case "nvidia-ai":
                from langchain_nvidia import ChatNVIDIA

                self.llm = ChatNVIDIA(
                    **llm_cfg,
                    stop="\n```",
                    temperature=config.llm.temperature,
                    max_completion_tokens=config.llm.max_tokens,
                )
            case "mistral-ai":
                from langchain_mistralai import ChatMistralAI

                self.llm = ChatMistralAI(
                    **llm_cfg,
                    stop=["\n```"],
                    max_tokens=config.llm.max_tokens,
                    temperature=config.llm.temperature,
                )
            case "deepseek-ai":
                from langchain_deepseek import ChatDeepSeek

                self.llm = ChatDeepSeek(
                    **llm_cfg,
                    max_tokens=config.llm.max_tokens,
                    temperature=config.llm.temperature,
                )
            case provider:
                raise ValueError(f"Model provider {provider} not supported")

        self.prompt_template: ChatPromptTemplate

    def __log_messages_stats(self, query: str):
        _logger.info("Sending query to model: %s", self.llm.name)
        num_chars = len(query)
        # num_tokens = len(tokenizer.encode(query))
        _logger.info("Query size: %s characters", num_chars)
        # _logger.info("Query size: %s characters (~%s tokens)", num_chars, num_tokens)

    def __update_llm_usage_statistics(self, query_at: float, usage: UsageMetadata | None):
        assert usage is not None

        self._num_llm_calls += 1
        self._time_calling_llm += time.time() - query_at
        self._input_tokens_cnt += usage["input_tokens"]
        self._output_tokens_cnt += usage["output_tokens"]

        _logger.info("Output size: %s tokens", usage["output_tokens"])

        stat.track_output_variable(RuntimeVariable.LLMCalls, self._num_llm_calls)
        stat.track_output_variable(RuntimeVariable.LLMQueryTime, self._time_calling_llm)
        stat.track_output_variable(RuntimeVariable.LLMInputTokens, self._input_tokens_cnt)
        stat.track_output_variable(RuntimeVariable.LLMOutputTokens, self._output_tokens_cnt)

    async def send_llm_request(
        self,
        *,
        query: str,
    ):
        query_at = time.time()
        self.__log_messages_stats(query)

        chain = self.prompt_template | self.llm
        response = await chain.ainvoke({"query": query})

        self.__update_llm_usage_statistics(query_at, response.usage_metadata)

        assert response.content is not None
        return response

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
                    f"# {f'TEST GENERATION BEGINS ({config.algorithm.name} + {self.llm.name} t={config.llm.temperature})':^64} #\n"
                    "####################################################################\n\n\n"
                )

            file.write(
                f"# {header} at query #{self._num_llm_calls}\n"
                "#--------------------------\n\n"
                f"{data}\n\n\n"
            )
