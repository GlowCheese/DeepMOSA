#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
import inspect
from typing import List

from libs.custom_logger import getLogger
from pynguin.configuration import Algorithm, config
from pynguin.llm.abstractmodel import AbstractLanguageModel, Messages
from pynguin.llm.codamosa.outputfixers import rewrite_tests
from pynguin.utils.generic import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)

logger = getLogger(__name__)


class _CodaMOSALanguageModel(AbstractLanguageModel):
    """Original language model implementation used by CodaMOSA"""

    @property
    def _system_prompt(self):
        return (
            "Write unit test for the given code object without any additional text or information.\n"
            "DO NOT include any import statement (assuming everything is correctly imported)."
        )

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

    def _call_completion(self, function_header: str, context_start: int, context_end: int):
        """Asks the model to provide a completion of the given function header,
        with the additional context of the target function definition.

        Args:
            function_header: a string containing a def statement to be completed
            context_start: the start line of context that must be included
            context_end: the end line of context that must be included

        Returns:
            the result of calling the model to complete the function header.
        """
        context = self._get_maximal_source_context(context_start, context_end)

        prompt = context + "\n" + function_header
        res = self.send_llm_request(
            [
                {"role": "system", "content": self._system_prompt},
                {"role": "assistant", "content": prompt},
            ],
            stop=["\n# Unit test for", "\ndef ", "\nclass "],
        )

        return prompt, res

    async def target_test_case(self, gao: GenericCallableAccessibleObject, context=""):
        """Provides a test case targeted to the function/method/constructor
        specified in `gao`

        Args:
            gao: a GenericCallableAccessibleObject to target the test to
            context: extra context to pass before the function header

        Returns:
            A generated test case as natural language
        """

        gao_desc: str
        test_signature: str

        if isinstance(gao, GenericMethod):
            gao_desc = f"method {gao.method_name} of class {gao.owner.name}"
            test_signature = f"def test_{gao.owner.name}_{gao.method_name}():"

            try:
                source_lines, start_line = inspect.getsourcelines(gao.owner.raw_type)
                end_line = start_line + len(source_lines) - 1
                if (
                    sum([self._get_num_tokens_at_line(i) for i in range(start_line, end_line + 1)])
                    > self._max_query_len
                ):
                    source_lines, start_line = inspect.getsourcelines(gao.owner.raw_type)
                    end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1

        elif isinstance(gao, GenericFunction):
            gao_desc = f"function {gao.function_name}"
            test_signature = f"def test_{gao.function_name}():"

            try:
                source_lines, start_line = inspect.getsourcelines(gao.callable)
                end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1

        elif isinstance(gao, GenericConstructor):
            class_name = gao.owner.name  # type: ignore
            gao_desc = f"constructor of class {class_name}"
            test_signature = f"def test_{class_name}():"

            try:
                source_lines, start_line = inspect.getsourcelines(
                    gao.generated_type().type.raw_type
                )
                end_line = start_line + len(source_lines)
            except (TypeError, OSError):
                start_line, end_line = -1, -1

        else:
            raise TypeError(f"Unsupported gao of type: {type(gao)}")

        context = self._get_maximal_source_context(start_line, end_line) + context

        prompt = (
            context
            + f"\nWrite unit test with pytest for {gao_desc} with the following signature: `{test_signature}`"
        )

        messages: Messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        # workaround for experimenting with DeepMOSA + codamosaseeding
        if config.algorithm != Algorithm.DEEPMOSA:
            response = self.send_llm_request(messages, stop="\n```")
        else:
            response = await self.send_llm_request_async(messages, stop="\n```")

        self._log_query_data("user_prompts.txt", prompt, "Prompt used")
        self._log_query_data("llm_raw_generated.py", response, "LLM-generated content")

        # Remove any trailing statements that don't parse
        generated_test = "\n".join(rewrite_tests(response).values())
        return generated_test


codamosalanguagemodel = _CodaMOSALanguageModel()

__all__ = ["codamosalanguagemodel"]
