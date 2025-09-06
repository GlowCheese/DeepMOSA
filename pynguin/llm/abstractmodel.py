import os
import time
from typing import List
from datetime import datetime
from abc import ABC, abstractmethod


from pynguin.globl import Globl
from vendor.deepseek import tokenizer
from pynguin.runtimevar import RuntimeVariable
from pynguin.deepmosa.datacenter import OpenAIDataCenter



class AbstractLanguageModel(ABC):
    """An interface for an OpenAI language model to generate/mutate tests as natural language."""

    def __init__(self):
        self.test_src: str
        self._max_query_len: int = 4000
        self._token_len_cache = {}
        self.num_codex_calls: int = 0
        self.time_calling_codex: float = 0
        self._datacenter = OpenAIDataCenter(Globl.conf.deepmosa.temperature, Globl.statistics_tracker)

    @abstractmethod
    def target_test_case(self, *args, **kwargs):
        pass

    def _get_maximal_source_context(
        self, start_line: int = -1, end_line: int = -1, used_tokens: int = 0
    ):
        """Tries to get the maximal source context that includes start_line to end_line but
        remains under the threshold.

        Args:
            start_line: the start line that should be included
            end_line: the end line that should be included
            used_tokens: the number of tokens to reduce the max allowed by

        Returns:
            as many lines from the source as possible that fit in max_context.
        """

        split_src = self.test_src.split("\n")
        num_lines = len(split_src)

        if end_line == -1:
            end_line = num_lines

        # Return everything if you can
        if (
            sum([self._get_num_tokens_at_line(i) for i in range(1, num_lines + 1)])
            < self._max_query_len
        ):
            return self.test_src

        if (
            sum([self._get_num_tokens_at_line(i) for i in range(1, end_line + 1)])
            < self._max_query_len
        ):
            return "\n".join(split_src[0:end_line])

        # Otherwise greedily take the lines preceding the end line
        cumul_len_of_prefix: List[int] = []
        cumul_len: int = 0
        for i in reversed(range(1, end_line + 1)):
            tok_len = self._get_num_tokens_at_line(i)
            cumul_len += tok_len
            cumul_len_of_prefix.insert(0, cumul_len)

        context_start_line = 0
        for idx, cumul_tok_len in enumerate(cumul_len_of_prefix):
            line_num = idx + 1
            if cumul_tok_len < self._max_query_len - used_tokens:
                context_start_line = line_num
                break

        return "\n".join(split_src[context_start_line:end_line])

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

    def record_llm_call(self, *, query_at: float):
        self.num_codex_calls += 1
        self.time_calling_codex += time.time() - query_at

        if (stat := Globl.statistics_tracker) is not None:
            stat.track_output_variable(RuntimeVariable.LLMCalls, self.num_codex_calls)
            stat.track_output_variable(RuntimeVariable.LLMQueryTime, self.time_calling_codex)

    def _log_prompt_used_and_response(
        self, prompt: str,
        raw_generated_test: str,
        generated_test_after_fixup: str
    ):
        """Log conversation and generated unit test for debugging purpose."""

        now = datetime.now()

        with open(
            os.path.join(Globl.report_dir, "gpt_raw_generated.py"),
            "a+", encoding="UTF-8"
        ) as log_file:
            log_file.write(f"\n\n# ({Globl.module_name}) Generated at {now}\n")
            log_file.write(raw_generated_test)

        with open(
            os.path.join(Globl.report_dir, "gpt_generated_after_fixup.py"),
            "a+", encoding="UTF-8"
        ) as log_file:
            log_file.write(f"\n\n# ({Globl.module_name}) Generated at {now}\n")
            log_file.write(generated_test_after_fixup)

        with open(
            os.path.join(Globl.report_dir, "gpt_prompts.py"),
            "a+", encoding="UTF-8",
        ) as log_file:
            log_file.write(f"\n\n# ({Globl.module_name}) prompt sent at {now}\n")
            log_file.write(prompt)
