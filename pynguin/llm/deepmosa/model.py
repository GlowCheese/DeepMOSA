#  This file is part of DeepMOSA.
#
#  SPDX-License-Identifier: MIT
#

from __future__ import annotations

import ast
import inspect
import os
import random
from collections import defaultdict
from typing import TYPE_CHECKING, Dict

from pynguin.configuration import config
from pynguin.llm.abstractmodel import AbstractLanguageModel
from pynguin.utils import randomness
from pynguin.utils.custom_logger import getLogger
from pynguin.utils.deepseek import tokenizer
from pynguin.utils.generic import (
    GenericCallableAccessibleObject,
    GenericConstructor,
    GenericFunction,
    GenericMethod,
)
from pynguin.utils.orderedset import OrderedSet

from .outputfixers import rewrite_tests

if TYPE_CHECKING:
    from pynguin.llm.abstractmodel import Messages

logger = getLogger(__name__)


class _DeepMOSALanguageModel(AbstractLanguageModel):
    """Language model implementation used by DeepMOSA"""

    def __init__(self):
        super().__init__()
        self._system_prompt = """
Do NOT import pytest and unittest when writting test cases.
A good unit test should only contains variable assignments, assertions and function/method/constructor calls (i.e. without any custom class or function definition or control structure like `if`, `for`, `while`, `match`, `with`, ... statements).
All test cases should starts with: `def test_[test case's name]():`.
Your response should only contain the test case itself without any additional text or information.
"""
        self._conversations: dict[GenericCallableAccessibleObject, Messages] = {}

    def _get_gao_str(self, gao: GenericCallableAccessibleObject):
        if not isinstance(gao, GenericCallableAccessibleObject):
            return None
        if not hasattr(gao._callable, "__code__"):
            return None
        try:
            assert os.path.isfile(gao.file_path)
            return inspect.getsource(gao._callable)
        except (TypeError, AssertionError, OSError):
            logger.debug("Cannot get source code for %s", gao._callable)
            return None

    def _get_annotated_gao_str(self, gao: GenericCallableAccessibleObject):
        source = self._get_gao_str(gao)
        if source is None:
            return None
        source = source.splitlines()
        pad = len(str(len(source)))
        for i in range(0, len(source)):
            source[i] = f"{i + 1:>{pad}} | {source[i]}"
        return "\n".join(source)

    def _safe_parse(self, source: str):
        original_source = source
        tmp = [line for line in source.splitlines() if line != "" and not line.isspace()]
        cnt = 0
        while tmp[0][cnt].isspace():
            cnt += 1
        source = "\n".join(line for line in tmp)

        try:
            if cnt == 0:
                return ast.parse(source)
            else:
                source = "if True:\n" + source
                result = ast.parse(source)
                assert isinstance(result, ast.Module)
                assert isinstance(result.body[0], ast.If)
                result.body = result.body[0].body
                return result
        except:
            logger.error(f"Original source:\n{original_source}")
            logger.error(f"Fixed source:\n{source}")
            raise

    def _take_until_full(self, source: str, focus_line=0, lim=None):
        if lim is None:
            lim = self._max_query_len

        source_list = source.splitlines()
        lo, hi = focus_line + 1, focus_line
        while lo - 1 >= 0 or hi + 1 < len(source_list):
            lo_len = len(tokenizer.encode(source_list[lo - 1]))
            hi_len = len(tokenizer.encode(source_list[hi + 1]))
            if lo > 0 and focus_line - lo <= hi - focus_line and lim - lo_len > 0:
                lo -= 1
                lim -= lo_len
            elif hi + 1 < len(source_list) and lim - hi_len > 0:
                hi += 1
                lim -= hi_len
            else:
                break

        if lo > hi:
            return ""
        else:
            return "\n".join(source_list[i] for i in range(lo, hi + 1))

    def _take_until_full_double_ends(
        self, source: str, end_1: int, end_2: int, lim: int | None = None
    ):
        if lim is None:
            lim = self._max_query_len

        source_list = source.splitlines()
        lo_1, hi_1, lim_1 = end_1 + 1, end_1, int(lim / 3)
        while lo_1 - 1 >= 0 or hi_1 + 1 < len(source_list):
            ok = False
            if lo_1 > 0 and end_1 - lo_1 <= hi_1 - end_1:
                lo_len = len(tokenizer.encode(source_list[lo_1 - 1]))
                if lim_1 - lo_len > 0:
                    lo_1 -= 1
                    lim_1 -= lo_len
                    ok = True

            if not ok and hi_1 + 1 < len(source_list):
                hi_len = len(tokenizer.encode(source_list[hi_1 + 1]))
                if lim_1 - hi_len > 0:
                    hi_1 += 1
                    lim_1 -= hi_len
                    ok = True

            if not ok:
                break

        lo_2, hi_2, lim_2 = end_2 + 1, end_2, int(2 * lim / 3)
        while lo_2 - 1 >= 0 or hi_2 + 1 < len(source_list):
            ok = False
            if lo_2 > 0 and end_2 - lo_2 <= hi_2 - end_2:
                lo_len = len(tokenizer.encode(source_list[lo_2 - 1]))
                if lim_2 - lo_len > 0:
                    lo_2 -= 1
                    lim_2 -= lo_len
                    ok = True

            if not ok and hi_2 + 1 < len(source_list):
                hi_len = len(tokenizer.encode(source_list[hi_2 + 1]))
                if lim_2 - hi_len > 0:
                    hi_2 += 1
                    lim_2 -= hi_len
                    ok = True

            if not ok:
                break

        if hi_1 + 1 >= lo_2:
            return self._take_until_full(source, end_1, lim) + "\n..."
        else:
            result_1 = "\n".join(source_list[i] for i in range(lo_1, hi_1 + 1))
            result_2 = "\n".join(source_list[i] for i in range(lo_2, hi_2 + 1))
            return result_1 + "\n...\n" + result_2

    def _get_maximal_source_context(
        self,
        gao: GenericCallableAccessibleObject,
        gao_owner_str: Dict[GenericCallableAccessibleObject, str],
        dependers: dict[
            GenericCallableAccessibleObject, OrderedSet[GenericCallableAccessibleObject]
        ],
        include_itself: bool,
        lim=None,
    ):
        if lim is None:
            lim = self._max_query_len

        # with open(gao.file_path, "r", encoding="utf-8") as file:
        #     module_tree = self._safe_parse(file.read())

        added_gaos = OrderedSet()
        gaos_map: dict[str, list[GenericCallableAccessibleObject]] = defaultdict(list)
        q = [gao]

        while len(q) > 0:
            new_q: OrderedSet[GenericCallableAccessibleObject] = OrderedSet()
            while len(q) > 0:
                selected_gao = q.pop(random.randrange(len(q)))
                if selected_gao in gao_owner_str:
                    selected_gao_str = gao_owner_str[selected_gao]
                else:
                    selected_gao_str = self._get_gao_str(selected_gao)
                if selected_gao_str is None:
                    continue

                curr_len = len(tokenizer.encode(selected_gao_str))
                if lim - curr_len > 0:
                    lim -= curr_len
                    added_gaos.add(selected_gao)
                    gaos_map[selected_gao.module_name].append(selected_gao)
                    new_q.update(dependers[selected_gao].difference(added_gaos))

            q = list(new_q)

        for mod, gao_set in gaos_map.items():
            gaos_map[mod] = gao_set[::-1]

        gao_module = gao.module_name

        def make_result(module_name: str) -> str:
            return "\n\n".join(
                gao_owner_str.get(x) or self._get_gao_str(x)  # type: ignore
                for x in gaos_map[module_name]
                if module_name != gao_module or include_itself is True or x in gao_owner_str
            )

        def module_to_path(module_name: str):
            module_name = module_name.replace(".", "/")
            return module_name + ".py"

        result = ""
        if len(gaos_map) == 0:
            # the length of the test object itself is too long!
            if gao.is_function():
                if include_itself:
                    result = self._take_until_full(self._get_gao_str(gao), lim=lim)  # type: ignore
                    result = f"```\n{result}\n```"
            elif gao in gao_owner_str:
                result = self._take_until_full(gao_owner_str[gao], lim=lim)
                result = f"```\n{result}\n```"

        # otherwise the length is sufficient, just take it easy
        elif len(gaos_map) == 1:
            result += make_result(gao_module)
            result = f"```\n{result}\n```"
        else:
            for mod in gaos_map.keys():
                if mod == gao_module:
                    continue
                result += module_to_path(mod) + ":\n"
                result += "```\n" + make_result(mod)
                result += "\n```\n\n"

            result += module_to_path(gao_module)
            result += " (module to test):\n```\n"
            result += make_result(gao_module) + "\n```"

        return result

    async def target_test_case(
        self,
        gao: GenericCallableAccessibleObject,
        gao_owner_str: Dict[GenericCallableAccessibleObject, str],
        dependers: dict[
            GenericCallableAccessibleObject, OrderedSet[GenericCallableAccessibleObject]
        ],
        pred_lineno: int | None,
        pred_value: bool | None,
    ) -> str:
        """Provides a test case targeted to the specified goal of the
        function/method/constructor specified in `gao`

        Returns:
            A generated test case as natural language.
        """

        messages: Messages | None = self._conversations.get(gao)
        gao_str = self._get_annotated_gao_str(gao)

        if (
            messages is None
            or gao_str is None
            or pred_lineno is None
            or randomness.chance(config.deepmosa.reseed_probability)
        ):
            if messages is None or randomness.chance(
                config.deepmosa.recreate_conversation_probability
            ):
                if isinstance(gao, GenericMethod):
                    instruction = (
                        f"Write unit test for method {gao.method_name} of class {gao.owner.name}"
                    )
                    try:
                        source_lines, start_line = inspect.getsourcelines(gao.owner.raw_type)
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
                            source_lines, start_line = inspect.getsourcelines(gao.owner.raw_type)
                            end_line = start_line + len(source_lines) - 1
                    except (TypeError, OSError):
                        start_line, end_line = -1, -1
                elif isinstance(gao, GenericFunction):
                    instruction = f"Write unit test for function {gao.function_name}"
                    try:
                        source_lines, start_line = inspect.getsourcelines(gao.callable)
                        end_line = start_line + len(source_lines) - 1
                    except (TypeError, OSError):
                        start_line, end_line = -1, -1
                elif isinstance(gao, GenericConstructor):
                    class_name = gao.owner.name  # type: ignore
                    instruction = f"Write unit test for the constructor of class {class_name}"
                    try:
                        source_lines, start_line = inspect.getsourcelines(
                            gao.generated_type().type.raw_type
                        )
                        end_line = start_line + len(source_lines)
                    except (TypeError, OSError):
                        start_line, end_line = -1, -1

                context = self._get_maximal_source_context(gao, gao_owner_str, dependers, True)
                messages = [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": f"{context}\n{instruction}"},
                ]
                self._conversations[gao] = messages

            response = await self.send_llm_request_async(messages, stop="\n```")

        else:
            if len(tokenizer.encode(gao_str)) >= self._max_query_len:
                context = ""
                gao_str = self._take_until_full_double_ends(
                    gao_str,
                    0,
                    pred_lineno - 1,
                    int(self._max_query_len / 3),  # is this necessary?
                )
            else:
                context = self._get_maximal_source_context(
                    gao,
                    gao_owner_str,
                    dependers,
                    False,
                    lim=self._max_query_len - len(tokenizer.encode(gao_str)),
                )
            instruction = (
                f"Write unit test to ensure that the predicate at "
                f"line {pred_lineno} evaluates to {pred_value}.\n"
                f"```\n{gao_str}\n```"
            )
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"{context}\n{instruction}"},
            ]
            response = await self.send_llm_request_async(messages, stop="\n```")

        # Remove any trailing statements that don't parse
        generated_test = "\n".join(rewrite_tests(response).values())
        self._log_prompt_used_and_response(messages[1]["content"], response, generated_test)
        return generated_test


deepmosalanguagemodel = _DeepMOSALanguageModel()

__all__ = ["deepmosalanguagemodel"]
