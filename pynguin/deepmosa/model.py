#  This file is part of CodaMOSA.
#
#  SPDX-FileCopyrightText: Microsoft
#
#  SPDX-License-Identifier: MIT
#
import ast
import inspect
import json
import os
import random
import time
from collections import defaultdict
from typing import Dict

from grappa import should

from vendor.custom_logger import getLogger
from pynguin.generic import (GenericCallableAccessibleObject,
                             GenericConstructor, GenericFunction,
                             GenericMethod)
from pynguin.globl import Globl
from pynguin.llm.abstractmodel import AbstractLanguageModel
from pynguin.utils import randomness
from vendor.deepseek import tokenizer
from vendor.orderedset import OrderedSet

from .api_errors import APIError, APIUnknownError
from .datacenter import CONVERSATION, DevPrompt
from .outputfixers import rewrite_tests

logger = getLogger(__name__)


class DeepMOSALanguageModel(AbstractLanguageModel):
    """Language model implementation used by DeepMOSA"""

    async def _call_completion(self, messages: CONVERSATION) -> str:
        """
        Send messages to the model and return its raw response.
        
        You need to use `json.loads` to convert it to `dict` manually.
        """
        query_at = time.time()

        try:
            resp = await self._datacenter.make_api_request(
                messages, DevPrompt.NO_CONTEXT
            )
            if resp.startswith('```py'):
                resp = resp.partition("\n")[2]
            elif resp.startswith('```'):
                resp = resp[3:]

            self.record_llm_call(query_at=query_at)
            return resp
                
        except APIUnknownError as e:
            logger.exception(e.message)
            exit(1)  # TODO: remove this line before production!
            return ""

        except APIError as e:
            logger.error("Failed to send message to GPT:\n%s", e.message)
            logger.error("Content of the messages sent to GPT:\n%s", json.dumps(messages))
            exit(17)  # TODO: remove this line before production!
            return ""


    def _get_gao_str(self, gao: GenericCallableAccessibleObject):
        if not isinstance(gao, GenericCallableAccessibleObject):
            return None
        if not hasattr(gao._callable, '__code__'):
            return None
        try:
            os.path.isfile(gao.file_path) | should.be.true
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
            source[i] = f"{i+1:>{pad}} | {source[i]}"
        return '\n'.join(source)


    def _safe_parse(self, source: str):
        original_source = source
        source = [
            line for line in source.splitlines()
            if line != '' and not line.isspace()
        ]
        cnt = 0
        while source[0][cnt].isspace(): cnt += 1
        source = '\n'.join(line for line in source)

        try:
            if cnt == 0:
                return ast.parse(source)
            else:
                source = "if True:\n" + source
                (result := ast.parse(source)) | should.be.a(ast.Module)
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
        while lo-1 >= 0 or hi+1 < len(source_list):
            lo_len = len(tokenizer.encode(source_list[lo-1]))
            hi_len = len(tokenizer.encode(source_list[hi+1]))
            if lo > 0 and focus_line-lo <= hi-focus_line and lim - lo_len > 0:
                lo -= 1
                lim -= lo_len
            elif hi+1 < len(source_list) and lim - hi_len > 0:
                hi += 1
                lim -= hi_len
            else:
                break

        if lo > hi:
            return ''
        else:
            return '\n'.join(
                source_list[i] for i in range(lo, hi+1)
            )
    

    def _take_until_full_double_ends(self, source: str, end_1: int, end_2: int, lim: None):
        if lim is None:
            lim = self._max_query_len
        
        source_list = source.splitlines()
        lo_1, hi_1, lim_1 = end_1 + 1, end_1, int(lim/3)
        while lo_1-1 >= 0 or hi_1+1 < len(source_list):
            ok = False
            if lo_1 > 0 and end_1 - lo_1 <= hi_1 - end_1:
                lo_len = len(tokenizer.encode(source_list[lo_1-1]))
                if lim_1 - lo_len > 0:
                    lo_1 -= 1
                    lim_1 -= lo_len
                    ok = True
            
            if not ok and hi_1+1 < len(source_list):
                hi_len = len(tokenizer.encode(source_list[hi_1+1]))
                if lim_1 - hi_len > 0:
                    hi_1 += 1
                    lim_1 -= hi_len
                    ok = True
            
            if not ok: break

        lo_2, hi_2, lim_2 = end_2 + 1, end_2, int(2*lim/3)
        while lo_2-1 >= 0 or hi_2+1 < len(source_list):
            ok = False
            if lo_2 > 0 and end_2-lo_2 <= hi_2-end_2:
                lo_len = len(tokenizer.encode(source_list[lo_2-1]))
                if lim_2 - lo_len > 0:
                    lo_2 -= 1
                    lim_2 -= lo_len
                    ok = True

            if not ok and hi_2+1 < len(source_list):
                hi_len = len(tokenizer.encode(source_list[hi_2+1]))
                if lim_2 - hi_len > 0:
                    hi_2 += 1
                    lim_2 -= hi_len
                    ok = True

            if not ok: break

        if hi_1 + 1 >= lo_2:
            return self._take_until_full(source, end_1, lim) + '\n...'
        else:
            result_1 = '\n'.join(source_list[i] for i in range(lo_1, hi_1+1))
            result_2 = '\n'.join(source_list[i] for i in range(lo_2, hi_2+1))
            return result_1 + '\n...\n' + result_2


    def _get_maximal_source_context(
        self,
        gao: GenericCallableAccessibleObject,
        gao_owner_str: Dict[GenericCallableAccessibleObject, str],
        dependers: dict[GenericCallableAccessibleObject,
            OrderedSet[GenericCallableAccessibleObject]],
        include_itself: bool,
        lim = None
    ):
        if lim is None: lim = self._max_query_len

        with open(gao.file_path, "r", encoding="utf-8") as file:
            module_tree = self._safe_parse(file.read())

        added_gaos = OrderedSet()
        gaos_map = defaultdict(OrderedSet)
        q = [gao]

        while len(q) > 0:
            new_q = OrderedSet()
            while len(q) > 0:
                selected_gao = q.pop(random.randrange(len(q)))
                if selected_gao in gao_owner_str:
                    selected_gao_str = gao_owner_str[selected_gao]
                else:
                    selected_gao_str = self._get_gao_str(selected_gao)
                if selected_gao_str is None: continue

                curr_len = len(tokenizer.encode(selected_gao_str))
                if lim - curr_len > 0:
                    lim -= curr_len
                    added_gaos.add(selected_gao)
                    gaos_map[selected_gao.module_name].add(selected_gao)
                    new_q.update(dependers[selected_gao].difference(added_gaos))
            
            q = list(new_q)

        for mod, gao_set in gaos_map.items():
            gaos_map[mod] = list(gao_set)[::-1]

        gao_module = gao.module_name

        def make_result(module_name):
            return '\n\n'.join(
                gao_owner_str.get(x)
                or self._get_gao_str(x)
                for x in gaos_map[module_name]
                if module_name != gao_module
                or include_itself is True
                or x in gao_owner_str
            )
        
        def module_to_path(module_name: str):
            module_name = module_name.replace('.', '/')
            return module_name + '.py'

        result = ""
        if len(gaos_map) == 0:
            # the length of the test object itself is too long!
            if gao.is_function():
                if include_itself:
                    result = self._take_until_full(
                        self._get_gao_str(gao), lim=lim)
                    result = f'```\n{result}\n```'
            elif gao in gao_owner_str:
                result = self._take_until_full(
                    gao_owner_str[gao], lim=lim)
                result = f'```\n{result}\n```'

        # otherwise the length is sufficient, just take it easy
        elif len(gaos_map) == 1:
            result += make_result(gao_module)
            result = f'```\n{result}\n```'
        else:
            for mod in gaos_map.keys():
                if mod == gao_module: continue
                result += module_to_path(mod) + ':\n'
                result += f'```\n' + make_result(mod)
                result += '\n```\n\n'

            result += module_to_path(gao_module)
            result += ' (module to test):\n```\n'
            result += make_result(gao_module) + '\n```'

        return result


    async def target_test_case(
        self,
        gao: GenericCallableAccessibleObject,
        gao_owner_str: Dict[GenericCallableAccessibleObject, str],
        dependers: dict[GenericCallableAccessibleObject,
            OrderedSet[GenericCallableAccessibleObject]],
        pred_lineno: int = None,
        pred_value: bool = None
    ) -> str:
        """Provides a test case targeted to the specified goal of the
        function/method/constructor specified in `gao`

        Returns:
            A generated test case as natural language.
        """

        conversation = self._datacenter[gao]
        gao_str = self._get_annotated_gao_str(gao)


        if (
            None in (conversation, gao_str, pred_lineno)
            or randomness.chance(Globl.conf.deepmosa.reseed_probability)
        ):
            if conversation is None or randomness.chance(
                Globl.conf.deepmosa.recreate_conversation_probability
            ):
                if gao.is_method():
                    method_gao: GenericMethod = gao
                    instruction = (
                        f"Write unit test for "
                        f"method {method_gao.method_name} "
                        f"of class {method_gao.owner.name}"
                    )
                    try:
                        source_lines, start_line = inspect.getsourcelines(method_gao.owner.raw_type)
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
                            source_lines, start_line = inspect.getsourcelines(method_gao.owner.raw_type)
                            end_line = start_line + len(source_lines) - 1
                    except (TypeError, OSError):
                        start_line, end_line = -1, -1
                elif gao.is_function():
                    fn_gao: GenericFunction = gao
                    instruction = (
                        f"Write unit test for "
                        f"function {fn_gao.function_name}"
                    )
                    try:
                        source_lines, start_line = inspect.getsourcelines(fn_gao.callable)
                        end_line = start_line + len(source_lines) - 1
                    except (TypeError, OSError):
                        start_line, end_line = -1, -1
                elif gao.is_constructor():
                    constructor_gao: GenericConstructor = gao  # type: ignore
                    class_name = constructor_gao.owner.name
                    instruction = (
                        f"Write unit test for "
                        f"the constructor of class {class_name}"
                    )
                    try:
                        source_lines, start_line = inspect.getsourcelines(
                            constructor_gao.generated_type().type.raw_type
                        )
                        end_line = start_line + len(source_lines)
                    except (TypeError, OSError):
                        start_line, end_line = -1, -1
                
                context = self._get_maximal_source_context(
                    gao, gao_owner_str, dependers, True
                )
                conversation: CONVERSATION = [
                    {'role': 'user', 'content': f'{context}\n{instruction}'}
                ]
                self._datacenter[gao] = conversation

            response = await self._call_completion(conversation)

        else:
            gao_str | should.not_be.none
            
            if len(tokenizer.encode(gao_str)) >= self._max_query_len:
                context = ''
                gao_str = self._take_until_full_double_ends(
                    gao_str, 0, pred_lineno-1,
                    self._max_query_len / 3  # is this necessary?
                )
            else:
                context = self._get_maximal_source_context(
                    gao, gao_owner_str, dependers, False,
                    lim=self._max_query_len-len(tokenizer.encode(gao_str))
                )
            instruction = (
                f"Write unit test to ensure that the predicate at "
                f"line {pred_lineno} evaluates to {pred_value}.\n"
                f"```\n{gao_str}\n```"
            )
            conversation: CONVERSATION = [
                {'role': 'user', 'content': f'{context}\n{instruction}'}
            ]
            response = await self._call_completion(conversation)
            

        # Remove any trailing statements that don't parse
        generated_test = '\n'.join(rewrite_tests(response).values())
        self._log_prompt_used_and_response(
            conversation[0]['content'], response, generated_test
        )
        return generated_test
