#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
import inspect
import time
from typing import Dict, List, cast

from pynguin.llm.abstractmodel import AbstractLanguageModel
from pynguin.llm.codamosa.outputfixers import fixup_result, rewrite_tests
from pynguin.utils.custom_logger import getLogger
from pynguin.utils.generic import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)

logger = getLogger(__name__)


class CodaMOSALanguageModel(AbstractLanguageModel):
    """Original language model implementation used by CodaMOSA"""

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

    def _call_completion(self, function_header: str, context_start: int, context_end: int) -> str:
        """Asks the model to provide a completion of the given function header,
        with the additional context of the target function definition.

        Args:
            function_header: a string containing a def statement to be completed
            context_start: the start line of context that must be included
            context_end: the end line of context that must be included

        Returns:
            the result of calling the model to complete the function header.
        """
        query_at = time.time()
        context = self._get_maximal_source_context(context_start, context_end)

        prompt = context + "\n" + function_header
        res = self.send_llm_request(
            [{"role": "assistant", "content": prompt, "prefix": True}],
            stop=["\n# Unit test for", "\ndef ", "\nclass "],
        )

        return res

    def target_test_case(self, gao: GenericCallableAccessibleObject, context="") -> str:
        """Provides a test case targeted to the function/method/constructor
        specified in `gao`

        Args:
            gao: a GenericCallableAccessibleObject to target the test to
            context: extra context to pass before the function header

        Returns:
            A generated test case as natural language.

        """
        if gao.is_method():
            method_gao = cast(GenericMethod, gao)
            function_header = (
                f"# Unit test for method {method_gao.method_name} of "
                f"class {method_gao.owner.name}\n"
                f"def test_{method_gao.owner.name}"
                f"_{method_gao.method_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(method_gao.owner.raw_type)
                end_line = start_line + len(source_lines) - 1
                if (
                    sum([self._get_num_tokens_at_line(i) for i in range(start_line, end_line + 1)])
                    > self._max_query_len
                ):
                    source_lines, start_line = inspect.getsourcelines(method_gao.owner.raw_type)  # type: ignore
                    end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1
        elif gao.is_function():
            fn_gao = cast(GenericFunction, gao)
            function_header = (
                f"# Unit test for function {fn_gao.function_name}"
                f"\ndef test_{fn_gao.function_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(fn_gao.callable)
                end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1
        elif gao.is_constructor():
            constructor_gao = cast(GenericConstructor, gao)
            class_name = constructor_gao.owner.name  # type: ignore
            function_header = (
                f"# Unit test for constructor of class {class_name}\ndef test_{class_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(
                    constructor_gao.generated_type().type.raw_type
                )
                end_line = start_line + len(source_lines)
            except (TypeError, OSError):
                start_line, end_line = -1, -1

        instruction = context + function_header
        response = self._call_completion(instruction, start_line, end_line)
        response = function_header + response  # type: ignore

        # Remove any trailing statements that don't parse
        generated_test = fixup_result(response)

        self._log_prompt_used_and_response(instruction, response, generated_test)

        generated_tests: Dict[str, str] = rewrite_tests(generated_test)
        for test_name in generated_tests:
            if test_name in function_header:
                return generated_tests[test_name]
        return ""
