#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
import inspect
import json
import os
import time
import requests
from typing import Dict
from types import SimpleNamespace
from custom_logger import getLogger

import environ

from pynguin.globl import Globl
from pynguin.codamosa.outputfixers import fixup_result, rewrite_tests
from pynguin.generic import (
    GenericMethod,
    GenericFunction,
    GenericConstructor,
    GenericCallableAccessibleObject,
)
from pynguin.llm.abstractmodel import AbstractLanguageModel

logger = getLogger(__name__)


def _openai_api_request(function_header, context):
    with open(
        os.path.join(Globl.report_dir, "codex_prompt.txt"),
        "a+",
        encoding="UTF-8",
    ) as log_file:
        log_file.write("\n\nPrompt written by CodaMOSA:\n")
        log_file.write(context + "\n" + function_header)
    
    payload = {
        "model": environ.OPENAI_COMP_MODEL,
        "prompt": context + "\n" + function_header,
        "temperature": Globl.conf.codamosa.temperature,
        "max_tokens": Globl.conf.codamosa.max_tokens,
        "stop": ["\n# Unit test for", "\ndef ", "\nclass "],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {environ.OPENAI_API_KEY}",
    }
    return payload, headers


class CodaMOSALanguageModel(AbstractLanguageModel):
    """Original language model implementation used by CodaMOSA"""

    @property
    def model_relative_url(self) -> str:
        if environ.DEFAULT_MODEL == "DEEPSEEK":
            return f"{environ.OPENAI_BASE_URL}/beta/completions"
        else:
            return f"{environ.OPENAI_BASE_URL}/completions"

    def _call_completion(
        self, function_header: str, context_start: int, context_end: int
    ) -> str:
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

        url = self.model_relative_url
        payload, headers = _openai_api_request(function_header, context)

        logger.info("Sending query using %s model", payload['model'])

        res = requests.post(url, data=json.dumps(payload), headers=headers)

        if res.status_code != 200:
            logger.error("Failed to call for completion:\n%s", res.json())
            return ""
        
        res = res.json()
        self._datacenter._record_query_usage(SimpleNamespace(**res['usage']))

        self.record_llm_call(query_at=query_at)
        return res['choices'][0]['text']


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
            method_gao: GenericMethod = gao  # type: ignore
            function_header = (
                f"# Unit test for method {method_gao.method_name} of "
                f"class {method_gao.owner.name}\n"  # type: ignore
                f"def test_{method_gao.owner.name}"
                f"_{method_gao.method_name}():"
            )
            try:
                source_lines, start_line = inspect.getsourcelines(method_gao.owner.raw_type)  # type: ignore
                end_line = start_line + len(source_lines) - 1
                if (
                    sum(
                        [
                            self._get_num_tokens_at_line(i)
                            for i in range(start_line, end_line + 1)
                        ]
                    )
                    > self._max_query_len
                ):
                    source_lines, start_line = inspect.getsourcelines(method_gao.owner.raw_type)  # type: ignore
                    end_line = start_line + len(source_lines) - 1
            except (TypeError, OSError):
                start_line, end_line = -1, -1
        elif gao.is_function():
            fn_gao: GenericFunction = gao  # type: ignore
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
            constructor_gao: GenericConstructor = gao  # type: ignore
            class_name = constructor_gao.owner.name
            function_header = (
                f"# Unit test for constructor of class {class_name}"
                f"\ndef test_{class_name}():"
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
        response = function_header + response

        # Remove any trailing statements that don't parse
        generated_test = fixup_result(response)

        # instruction = response = ''
        # with open(".trash/hint.py", "r") as file:
        #     generated_test = file.read()
        # generated_test = fixup_result(generated_test)

        self._log_prompt_used_and_response(instruction, response, generated_test)

        generated_tests: Dict[str, str] = rewrite_tests(generated_test)
        for test_name in generated_tests:
            if test_name in function_header:
                return generated_tests[test_name]
        return ""
