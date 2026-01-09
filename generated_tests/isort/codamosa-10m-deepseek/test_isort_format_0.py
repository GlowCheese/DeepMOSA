# Check out: https://github.com/GlowCheese/deepmosa
import re as module_2

import colorama.ansi as module_3
import colorama.initialise as module_1
import isort.format as module_0
import pytest


def test_case_0():
    var_0 = '0 (#'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == '0 (#'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = "9Z0'1?D;i/q{k\n*"
    var_3 = module_0.remove_whitespace(var_2)
    assert var_3 == "9Z0'1?D;i/q{k*"

def test_case_1():
    var_0 = '\x0cpa'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import pa'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'D8==c3ZT3.'
    var_1 = '=T8d]7*6BMPc<\nQt/[<'
    var_2 = None
    var_3 = module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_2)
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = module_0.BasicPrinter(var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message is None
    assert var_4.error_message is None
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5 = '+G;CRGkdW5"s0>s_tX'
    var_4.error(var_5)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'DIZ(tbUDKQxjC10%\r'
    module_0.ask_whether_to_apply_changes_to_file(var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = True
    var_1 = None
    var_2 = module_0.create_terminal_printer(var_0, var_0, success=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert var_2.output is True
    assert var_2.success_message is None
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = module_1.just_fix_windows_console()
    assert f'{type(module_1.orig_stdout).__module__}.{type(module_1.orig_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.orig_stdout.mode == 'w'
    assert f'{type(module_1.orig_stderr).__module__}.{type(module_1.orig_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.orig_stderr.mode == 'w'
    assert f'{type(module_1.wrapped_stdout).__module__}.{type(module_1.wrapped_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.wrapped_stdout.mode == 'w'
    assert f'{type(module_1.wrapped_stderr).__module__}.{type(module_1.wrapped_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.wrapped_stderr.mode == 'w'
    assert module_1.atexit_done is True
    assert module_1.fixed_windows_console is False
    var_3.__exit__()

def test_case_6():
    var_0 = True
    var_1 = module_0.create_terminal_printer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message == ''
    assert var_1.error_message == ''
    assert var_1.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_1.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_1.ADDED_LINE == '\x1b[32m'
    assert var_1.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.remove_whitespace(var_0, var_0)

def test_case_8():
    pass

def test_case_9():
    var_0 = True
    var_1 = None
    var_2 = module_0.create_terminal_printer(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = None
    var_4 = var_2.success(var_3)
    var_5 = module_0.create_terminal_printer(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_5.output).__module__}.{type(var_5.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_5.success_message == ''
    assert var_5.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '(,q^a<G 7$]K6eNKX<5'
    var_1 = module_0.BasicPrinter(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message == '(,q^a<G 7$]K6eNKX<5'
    assert var_1.error_message == '(,q^a<G 7$]K6eNKX<5'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_2 = None
    var_1.diff_line(var_2)

def test_case_11():
    var_0 = "7h~'z!Z.)\r"
    var_1 = module_0.format_natural(var_0)
    assert var_1 == "from 7h~'z!Z import )"
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = None
    var_3 = None
    var_4 = module_0.create_terminal_printer(var_3, var_2, success=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message is None
    assert var_4.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = True
    var_2 = module_0.create_terminal_printer(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = "z'\x0bUfj"
    var_4 = module_0.create_terminal_printer(var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == ''
    assert var_4.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5 = var_2.diff_line(var_3)
    var_6 = 'r'
    var_7 = False
    module_0.show_unified_diff(file_input=var_5, file_output=var_6, file_path=var_5, color_output=var_7)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = "z'E\x0bU\\fyj"
    var_1 = True
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0, color_output=var_1)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 'W_*IJh6W'
    var_2 = "z'E\x0bU\\fyj"
    var_3 = '9HH+'
    var_4 = True
    var_5 = module_0.show_unified_diff(file_input=var_2, file_output=var_3, file_path=var_0, color_output=var_4)
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_6 = module_0.format_natural(var_1)
    assert var_6 == 'import W_*IJh6W'
    module_2.template(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.create_terminal_printer(var_0, success=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message is None
    assert var_1.error_message == ''
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_2 = "z'E\x0bU\\fyj"
    var_3 = True
    var_4 = module_0.create_terminal_printer(var_0, error=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == ''
    assert var_4.error_message == "z'E\x0bU\\fyj"
    var_5 = module_1.reinit()
    assert f'{type(module_1.orig_stdout).__module__}.{type(module_1.orig_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.orig_stdout.mode == 'w'
    assert f'{type(module_1.orig_stderr).__module__}.{type(module_1.orig_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.orig_stderr.mode == 'w'
    assert f'{type(module_1.wrapped_stdout).__module__}.{type(module_1.wrapped_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.wrapped_stdout.mode == 'w'
    assert f'{type(module_1.wrapped_stderr).__module__}.{type(module_1.wrapped_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_1.wrapped_stderr.mode == 'w'
    assert module_1.atexit_done is True
    assert module_1.fixed_windows_console is False
    var_6 = module_0.show_unified_diff(file_input=var_2, file_output=var_2, file_path=var_0, color_output=var_3)
    var_7 = None
    var_8 = module_3.AnsiBack()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'colorama.ansi.AnsiBack'
    assert var_8.BLACK == '\x1b[40m'
    assert var_8.BLUE == '\x1b[44m'
    assert var_8.CYAN == '\x1b[46m'
    assert var_8.GREEN == '\x1b[42m'
    assert var_8.LIGHTBLACK_EX == '\x1b[100m'
    assert var_8.LIGHTBLUE_EX == '\x1b[104m'
    assert var_8.LIGHTCYAN_EX == '\x1b[106m'
    assert var_8.LIGHTGREEN_EX == '\x1b[102m'
    assert var_8.LIGHTMAGENTA_EX == '\x1b[105m'
    assert var_8.LIGHTRED_EX == '\x1b[101m'
    assert var_8.LIGHTWHITE_EX == '\x1b[107m'
    assert var_8.LIGHTYELLOW_EX == '\x1b[103m'
    assert var_8.MAGENTA == '\x1b[45m'
    assert var_8.RED == '\x1b[41m'
    assert var_8.RESET == '\x1b[49m'
    assert var_8.WHITE == '\x1b[47m'
    assert var_8.YELLOW == '\x1b[43m'
    assert module_3.CSI == '\x1b['
    assert module_3.OSC == '\x1b]'
    assert module_3.BEL == '\x07'
    assert f'{type(module_3.Fore).__module__}.{type(module_3.Fore).__qualname__}' == 'colorama.ansi.AnsiFore'
    assert module_3.Fore.BLACK == '\x1b[30m'
    assert module_3.Fore.BLUE == '\x1b[34m'
    assert module_3.Fore.CYAN == '\x1b[36m'
    assert module_3.Fore.GREEN == '\x1b[32m'
    assert module_3.Fore.LIGHTBLACK_EX == '\x1b[90m'
    assert module_3.Fore.LIGHTBLUE_EX == '\x1b[94m'
    assert module_3.Fore.LIGHTCYAN_EX == '\x1b[96m'
    assert module_3.Fore.LIGHTGREEN_EX == '\x1b[92m'
    assert module_3.Fore.LIGHTMAGENTA_EX == '\x1b[95m'
    assert module_3.Fore.LIGHTRED_EX == '\x1b[91m'
    assert module_3.Fore.LIGHTWHITE_EX == '\x1b[97m'
    assert module_3.Fore.LIGHTYELLOW_EX == '\x1b[93m'
    assert module_3.Fore.MAGENTA == '\x1b[35m'
    assert module_3.Fore.RED == '\x1b[31m'
    assert module_3.Fore.RESET == '\x1b[39m'
    assert module_3.Fore.WHITE == '\x1b[37m'
    assert module_3.Fore.YELLOW == '\x1b[33m'
    assert f'{type(module_3.Back).__module__}.{type(module_3.Back).__qualname__}' == 'colorama.ansi.AnsiBack'
    assert module_3.Back.BLACK == '\x1b[40m'
    assert module_3.Back.BLUE == '\x1b[44m'
    assert module_3.Back.CYAN == '\x1b[46m'
    assert module_3.Back.GREEN == '\x1b[42m'
    assert module_3.Back.LIGHTBLACK_EX == '\x1b[100m'
    assert module_3.Back.LIGHTBLUE_EX == '\x1b[104m'
    assert module_3.Back.LIGHTCYAN_EX == '\x1b[106m'
    assert module_3.Back.LIGHTGREEN_EX == '\x1b[102m'
    assert module_3.Back.LIGHTMAGENTA_EX == '\x1b[105m'
    assert module_3.Back.LIGHTRED_EX == '\x1b[101m'
    assert module_3.Back.LIGHTWHITE_EX == '\x1b[107m'
    assert module_3.Back.LIGHTYELLOW_EX == '\x1b[103m'
    assert module_3.Back.MAGENTA == '\x1b[45m'
    assert module_3.Back.RED == '\x1b[41m'
    assert module_3.Back.RESET == '\x1b[49m'
    assert module_3.Back.WHITE == '\x1b[47m'
    assert module_3.Back.YELLOW == '\x1b[43m'
    assert f'{type(module_3.Style).__module__}.{type(module_3.Style).__qualname__}' == 'colorama.ansi.AnsiStyle'
    assert module_3.Style.BRIGHT == '\x1b[1m'
    assert module_3.Style.DIM == '\x1b[2m'
    assert module_3.Style.NORMAL == '\x1b[22m'
    assert module_3.Style.RESET_ALL == '\x1b[0m'
    assert f'{type(module_3.Cursor).__module__}.{type(module_3.Cursor).__qualname__}' == 'colorama.ansi.AnsiCursor'
    assert module_3.AnsiBack.BLACK == 40
    assert module_3.AnsiBack.RED == 41
    assert module_3.AnsiBack.GREEN == 42
    assert module_3.AnsiBack.YELLOW == 43
    assert module_3.AnsiBack.BLUE == 44
    assert module_3.AnsiBack.MAGENTA == 45
    assert module_3.AnsiBack.CYAN == 46
    assert module_3.AnsiBack.WHITE == 47
    assert module_3.AnsiBack.RESET == 49
    assert module_3.AnsiBack.LIGHTBLACK_EX == 100
    assert module_3.AnsiBack.LIGHTRED_EX == 101
    assert module_3.AnsiBack.LIGHTGREEN_EX == 102
    assert module_3.AnsiBack.LIGHTYELLOW_EX == 103
    assert module_3.AnsiBack.LIGHTBLUE_EX == 104
    assert module_3.AnsiBack.LIGHTMAGENTA_EX == 105
    assert module_3.AnsiBack.LIGHTCYAN_EX == 106
    assert module_3.AnsiBack.LIGHTWHITE_EX == 107
    var_9 = "7h~'z!Z.)\r"
    var_10 = module_0.format_natural(var_9)
    assert var_10 == "from 7h~'z!Z import )"
    var_11 = module_0.format_natural(var_10)
    assert var_11 == "from 7h~'z!Z import )"
    var_5.__getitem__(var_7, var_5)

def test_case_16():
    var_0 = 'import module'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == 'module'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = 'import module.submodule'
    var_3 = module_0.format_simplified(var_2)
    assert var_3 == 'module.submodule'
    var_4 = 'from module.submodule import name'
    var_5 = module_0.format_simplified(var_4)
    assert var_5 == 'module.submodule.name'
    var_6 = '  from module import name  '
    var_7 = module_0.format_simplified(var_6)
    assert var_7 == 'module.name'
    var_8 = '  import module.submodule  '
    var_9 = module_0.format_simplified(var_8)
    assert var_9 == 'module.submodule'
    var_10 = '  from module.submodule import name  '
    var_11 = module_0.format_simplified(var_10)
    assert var_11 == 'module.submodule.name'
    var_12 = 'All tests passed for format_simplified'
    var_13 = print(var_12)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'import os'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import os'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = None
    module_0.ask_whether_to_apply_changes_to_file(var_2)