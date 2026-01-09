# Check out: https://github.com/GlowCheese/deepmosa
import pathlib as module_0

import colorama.ansi as module_2
import colorama.initialise as module_3
import isort.format as module_1
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.Path
    var_1 = 'O#\t>r2F3'
    var_2 = module_1.format_simplified(var_1)
    assert var_2 == 'O#\t>r2F3'
    assert module_0.EINVAL == 22
    assert module_0.ENOENT == 2
    assert module_0.ENOTDIR == 20
    assert module_0.EBADF == 9
    assert module_0.ELOOP == 40
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    module_2.set_title(var_0)

def test_case_1():
    var_0 = '*&'
    var_1 = module_1.format_natural(var_0)
    assert var_1 == 'import *&'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_1.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = False
    module_1.ask_whether_to_apply_changes_to_file(var_0)

def test_case_4():
    var_0 = None
    var_1 = module_1.create_terminal_printer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message == ''
    assert var_1.error_message == ''
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_1.BasicPrinter.ERROR == 'ERROR'
    assert module_1.BasicPrinter.SUCCESS == 'SUCCESS'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = ']\\7gQp?'
    var_1 = '1+xYM6'
    var_2 = module_1.format_natural(var_1)
    assert var_2 == 'import 1+xYM6'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = None
    var_4 = 'wSXVw81-1=>+p*\x0ba\\O'
    var_5 = True
    var_6 = module_1.create_terminal_printer(var_5, success=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message is None
    assert var_6.error_message == ''
    assert var_6.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_6.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_6.ADDED_LINE == '\x1b[32m'
    assert var_6.REMOVED_LINE == '\x1b[31m'
    var_7 = module_1.format_simplified(var_4)
    assert var_7 == 'wSXVw81-1=>+p*\x0ba\\O'
    var_8 = '8{'
    var_9 = module_1.show_unified_diff(file_input=var_8, file_output=var_8, file_path=var_3, color_output=var_5)
    var_10 = var_6.diff_line(var_0)
    module_1.ask_whether_to_apply_changes_to_file(var_3)

def test_case_6():
    var_0 = True
    var_1 = None
    var_2 = module_1.create_terminal_printer(var_0, error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message is None
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_1.remove_whitespace(var_0)

def test_case_8():
    pass

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'p4QD'
    var_1 = None
    var_2 = module_1.ColoramaPrinter(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message is None
    assert var_2.error_message is None
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2.error(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'xC_\x0b)mY=Z+lvw\\T'
    var_1 = None
    var_2 = module_1.BasicPrinter(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message is None
    assert var_2.error_message is None
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_1.BasicPrinter.ERROR == 'ERROR'
    assert module_1.BasicPrinter.SUCCESS == 'SUCCESS'
    var_2.success(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = ']\\7gQp?'
    var_1 = module_1.format_simplified(var_0)
    assert var_1 == ']\\7gQp?'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = '1+xYM6'
    var_3 = module_1.format_natural(var_2)
    assert var_3 == 'import 1+xYM6'
    var_4 = None
    var_5 = True
    var_6 = module_1.create_terminal_printer(var_5, success=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message is None
    assert var_6.error_message == ''
    assert var_6.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_6.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_6.ADDED_LINE == '\x1b[32m'
    assert var_6.REMOVED_LINE == '\x1b[31m'
    var_7 = '\x0bO9Jn((KR$R8'
    var_8 = '8'
    var_9 = module_1.show_unified_diff(file_input=var_8, file_output=var_7, file_path=var_4, color_output=var_5)
    module_1.remove_whitespace(var_5)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = False
    var_2 = module_1.create_terminal_printer(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message == ''
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_1.BasicPrinter.ERROR == 'ERROR'
    assert module_1.BasicPrinter.SUCCESS == 'SUCCESS'
    var_2.diff_line(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = module_3.just_fix_windows_console()
    assert f'{type(module_3.orig_stdout).__module__}.{type(module_3.orig_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_3.orig_stdout.mode == 'w'
    assert f'{type(module_3.orig_stderr).__module__}.{type(module_3.orig_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_3.orig_stderr.mode == 'w'
    assert f'{type(module_3.wrapped_stdout).__module__}.{type(module_3.wrapped_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_3.wrapped_stdout.mode == 'w'
    assert f'{type(module_3.wrapped_stderr).__module__}.{type(module_3.wrapped_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_3.wrapped_stderr.mode == 'w'
    assert module_3.atexit_done is True
    assert module_3.fixed_windows_console is False
    var_2 = '\x0c~IK(iV{`_\r'
    var_3 = module_1.format_natural(var_2)
    assert var_3 == 'import ~IK(iV{`_'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = 'wSXVw81-1=>+p*\x0ba\\O'
    var_5 = module_1.create_terminal_printer(var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.BasicPrinter'
    assert var_5.output == 'wSXVw81-1=>+p*\x0ba\\O'
    assert var_5.success_message == ''
    assert var_5.error_message == ''
    assert module_1.BasicPrinter.ERROR == 'ERROR'
    assert module_1.BasicPrinter.SUCCESS == 'SUCCESS'
    var_6 = 'jB\\`'
    var_7 = module_1.remove_whitespace(var_6)
    assert var_7 == 'jB\\`'
    var_1.set_title(var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = ']\\7gQp?'
    var_1 = module_1.format_simplified(var_0)
    assert var_1 == ']\\7gQp?'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = '?q~4'
    var_3 = module_1.create_terminal_printer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_3.output).__module__}.{type(var_3.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_3.success_message == ''
    assert var_3.error_message == ''
    assert var_3.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_3.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_3.ADDED_LINE == '\x1b[32m'
    assert var_3.REMOVED_LINE == '\x1b[31m'
    module_1.show_unified_diff(file_input=var_1, file_output=var_2, file_path=var_1)

def test_case_15():
    var_0 = 'import os'
    var_1 = module_1.format_natural(var_0)
    assert var_1 == 'import os'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = module_1.format_natural(var_1)
    assert var_2 == 'import os'
    var_3 = 'os.path'
    var_4 = module_1.format_natural(var_3)
    assert var_4 == 'from os import path'
    var_5 = 'os'
    var_6 = module_1.format_natural(var_5)
    assert var_6 == 'import os'
    var_7 = 'os.path.join'
    var_8 = module_1.format_natural(var_7)
    assert var_8 == 'from os.path import join'
    var_9 = '  os  '
    var_10 = module_1.format_natural(var_9)
    assert var_10 == 'import os'
    var_11 = '  from os import pHth  '
    var_12 = module_1.format_natural(var_11)
    assert var_12 == 'from os import pHth'
    var_13 = ''
    var_14 = module_1.format_natural(var_13)
    assert var_14 == 'import '
    var_15 = 'os.path.join.split'
    var_16 = module_1.format_natural(var_15)
    assert var_16 == 'from os.path.join import split'
    var_17 = 'os.path.join.split.strip'
    var_18 = module_1.format_natural(var_17)
    assert var_18 == 'from os.path.join.split import strip'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'import os'
    var_1 = module_1.format_simplified(var_0)
    assert var_1 == 'os'
    assert module_1.colorama_unavailable is False
    assert f'{type(module_1.ADDED_LINE_PATTERN).__module__}.{type(module_1.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_1.REMOVED_LINE_PATTERN).__module__}.{type(module_1.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = 'from os import path'
    var_3 = module_1.format_simplified(var_2)
    assert var_3 == 'os.path'
    var_4 = module_1.format_simplified(var_1)
    assert var_4 == 'os'
    var_5 = 'import os.path'
    var_6 = module_1.format_simplified(var_5)
    assert var_6 == 'os.path'
    var_7 = 'import os.path as osp'
    var_8 = module_1.format_simplified(var_7)
    assert var_8 == 'os.path as osp'
    var_9 = 'from os.path import join as j'
    var_10 = module_1.format_simplified(var_9)
    assert var_10 == 'os.path.join as j'
    var_11 = 'from os.path import join as j, split as s'
    var_12 = module_1.format_simplified(var_11)
    assert var_12 == 'os.path.join as j, split as s'
    var_13 = 'from os.path import join as j, split as s, abspath as a'
    var_14 = module_1.format_simplified(var_13)
    assert var_14 == 'os.path.join as j, split as s, abspath as a'
    var_15 = 'from os.path import join as j, split as s, abspath as a, dirname as d'
    var_16 = module_1.format_simplified(var_15)
    assert var_16 == 'os.path.join as j, split as s, abspath as a, dirname as d'
    var_17 = 'from os.path import join as j, split as s, abspath as a, dirname as d, basename as b'
    var_18 = module_1.format_simplified(var_17)
    assert var_18 == 'os.path.join as j, split as s, abspath as a, dirname as d, basename as b'
    var_19 = 'from os.path import join as j, split as s, abspath as a, dirname as d, basename as b, isfile as i'
    var_20 = module_1.format_simplified(var_19)
    assert var_20 == 'os.path.join as j, split as s, abspath as a, dirname as d, basename as b, isfile as i'
    var_21 = None
    module_1.ask_whether_to_apply_changes_to_file(var_21)