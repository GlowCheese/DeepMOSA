# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.format as module_0
import colorama.ansi as module_1
import colorama.initialise as module_2

def test_case_0():
    var_0 = None
    var_1 = '\x0b,VbFB^yTXhM'
    var_2 = module_0.format_simplified(var_1)
    assert var_2 == ',VbFB^yTXhM'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = 'N~{+'
    var_4 = module_0.BasicPrinter(var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == 'N~{+'
    assert var_4.error_message is None
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.create_terminal_printer(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message == ''
    assert var_1.error_message == ''
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0)

def test_case_2():
    var_0 = None
    var_1 = '\nf[fC;'
    var_2 = module_0.format_simplified(var_1)
    assert var_2 == 'f[fC;'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = module_0.format_natural(var_1)
    assert var_3 == 'import f[fC;'
    var_4 = ''
    var_5 = 'V[y#\x0cB<P\n'
    var_6 = True
    var_7 = module_0.show_unified_diff(file_input=var_4, file_output=var_5, file_path=var_0, output=var_0, color_output=var_6)
    var_8 = module_1.code_to_chars(var_0)
    assert var_8 == '\x1b[Nonem'
    assert module_1.CSI == '\x1b['
    assert module_1.OSC == '\x1b]'
    assert module_1.BEL == '\x07'
    assert f'{type(module_1.Fore).__module__}.{type(module_1.Fore).__qualname__}' == 'colorama.ansi.AnsiFore'
    assert module_1.Fore.BLACK == '\x1b[30m'
    assert module_1.Fore.BLUE == '\x1b[34m'
    assert module_1.Fore.CYAN == '\x1b[36m'
    assert module_1.Fore.GREEN == '\x1b[32m'
    assert module_1.Fore.LIGHTBLACK_EX == '\x1b[90m'
    assert module_1.Fore.LIGHTBLUE_EX == '\x1b[94m'
    assert module_1.Fore.LIGHTCYAN_EX == '\x1b[96m'
    assert module_1.Fore.LIGHTGREEN_EX == '\x1b[92m'
    assert module_1.Fore.LIGHTMAGENTA_EX == '\x1b[95m'
    assert module_1.Fore.LIGHTRED_EX == '\x1b[91m'
    assert module_1.Fore.LIGHTWHITE_EX == '\x1b[97m'
    assert module_1.Fore.LIGHTYELLOW_EX == '\x1b[93m'
    assert module_1.Fore.MAGENTA == '\x1b[35m'
    assert module_1.Fore.RED == '\x1b[31m'
    assert module_1.Fore.RESET == '\x1b[39m'
    assert module_1.Fore.WHITE == '\x1b[37m'
    assert module_1.Fore.YELLOW == '\x1b[33m'
    assert f'{type(module_1.Back).__module__}.{type(module_1.Back).__qualname__}' == 'colorama.ansi.AnsiBack'
    assert module_1.Back.BLACK == '\x1b[40m'
    assert module_1.Back.BLUE == '\x1b[44m'
    assert module_1.Back.CYAN == '\x1b[46m'
    assert module_1.Back.GREEN == '\x1b[42m'
    assert module_1.Back.LIGHTBLACK_EX == '\x1b[100m'
    assert module_1.Back.LIGHTBLUE_EX == '\x1b[104m'
    assert module_1.Back.LIGHTCYAN_EX == '\x1b[106m'
    assert module_1.Back.LIGHTGREEN_EX == '\x1b[102m'
    assert module_1.Back.LIGHTMAGENTA_EX == '\x1b[105m'
    assert module_1.Back.LIGHTRED_EX == '\x1b[101m'
    assert module_1.Back.LIGHTWHITE_EX == '\x1b[107m'
    assert module_1.Back.LIGHTYELLOW_EX == '\x1b[103m'
    assert module_1.Back.MAGENTA == '\x1b[45m'
    assert module_1.Back.RED == '\x1b[41m'
    assert module_1.Back.RESET == '\x1b[49m'
    assert module_1.Back.WHITE == '\x1b[47m'
    assert module_1.Back.YELLOW == '\x1b[43m'
    assert f'{type(module_1.Style).__module__}.{type(module_1.Style).__qualname__}' == 'colorama.ansi.AnsiStyle'
    assert module_1.Style.BRIGHT == '\x1b[1m'
    assert module_1.Style.DIM == '\x1b[2m'
    assert module_1.Style.NORMAL == '\x1b[22m'
    assert module_1.Style.RESET_ALL == '\x1b[0m'
    assert f'{type(module_1.Cursor).__module__}.{type(module_1.Cursor).__qualname__}' == 'colorama.ansi.AnsiCursor'

def test_case_3():
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

def test_case_4():
    var_0 = '`N\x0c@'
    var_1 = None
    var_2 = 'waZ'
    var_3 = module_0.remove_whitespace(var_2)
    assert var_3 == 'waZ'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = 'udvKH2O9zhFN'
    var_5 = module_0.ColoramaPrinter(var_1, var_4, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_5.output).__module__}.{type(var_5.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_5.success_message == 'udvKH2O9zhFN'
    assert var_5.error_message is None
    assert var_5.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_5.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_5.ADDED_LINE == '\x1b[32m'
    assert var_5.REMOVED_LINE == '\x1b[31m'
    var_6 = var_5.style_text(var_0, var_0)
    assert var_6 == '`N\x0c@`N\x0c@\x1b[0m'

def test_case_5():
    var_0 = '\nf[fC;'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import f[fC;'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'p4QD'
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message is None
    assert var_2.error_message is None
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2.error(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_2.colorama_text()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'contextlib._GeneratorContextManager'
    assert f'{type(var_1.gen).__module__}.{type(var_1.gen).__qualname__}' == 'builtins.generator'
    assert var_1.args == ()
    assert var_1.kwds == {}
    assert f'{type(module_2.orig_stdout).__module__}.{type(module_2.orig_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.orig_stdout.mode == 'w'
    assert f'{type(module_2.orig_stderr).__module__}.{type(module_2.orig_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.orig_stderr.mode == 'w'
    assert f'{type(module_2.wrapped_stdout).__module__}.{type(module_2.wrapped_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.wrapped_stdout.mode == 'w'
    assert f'{type(module_2.wrapped_stderr).__module__}.{type(module_2.wrapped_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.wrapped_stderr.mode == 'w'
    assert module_2.atexit_done is True
    assert module_2.fixed_windows_console is False
    var_2 = 'btPC(#K4\r'
    var_3 = 'iBCYH[u-'
    var_4 = module_0.ColoramaPrinter(var_3, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert var_4.output == 'btPC(#K4\r'
    assert var_4.success_message == 'btPC(#K4\r'
    assert var_4.error_message == 'iBCYH[u-'
    assert var_4.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_4.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_4.ADDED_LINE == '\x1b[32m'
    assert var_4.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_1.diff_line(var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_2 = None
    var_3 = module_0.create_terminal_printer(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_3.output).__module__}.{type(var_3.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_3.success_message == ''
    assert var_3.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_4 = 'o#a0&;4zB5'
    var_5 = var_3.success(var_4)
    var_6 = module_2.just_fix_windows_console()
    assert f'{type(module_2.orig_stdout).__module__}.{type(module_2.orig_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.orig_stdout.mode == 'w'
    assert f'{type(module_2.orig_stderr).__module__}.{type(module_2.orig_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.orig_stderr.mode == 'w'
    assert f'{type(module_2.wrapped_stdout).__module__}.{type(module_2.wrapped_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.wrapped_stdout.mode == 'w'
    assert f'{type(module_2.wrapped_stderr).__module__}.{type(module_2.wrapped_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.wrapped_stderr.mode == 'w'
    assert module_2.atexit_done is True
    assert module_2.fixed_windows_console is False
    var_7 = '( \t=7"R'
    module_0.remove_whitespace(var_2, var_7)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = '@'
    var_2 = ',a(Dzm 9zCy\\!'
    var_3 = module_0.show_unified_diff(file_input=var_1, file_output=var_2, file_path=var_0, color_output=var_0)
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = module_0.remove_whitespace(var_2)
    assert var_4 == ',a(Dzm9zCy\\!'
    var_5 = module_0.create_terminal_printer(var_0, success=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_5.output).__module__}.{type(var_5.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_5.success_message is None
    assert var_5.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5.diff_line(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = 'd\tZ7?xI.Ao~drRm4AE\r'
    var_2 = module_0.format_natural(var_1)
    assert var_2 == 'from d\tZ7?xI import Ao~drRm4AE'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = '\nf[fC;'
    var_2 = module_0.format_simplified(var_1)
    assert var_2 == 'f[fC;'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = module_0.format_natural(var_1)
    assert var_3 == 'import f[fC;'
    var_4 = 'V[y#\x0cB<P\n'
    var_5 = True
    var_6 = module_0.show_unified_diff(file_input=var_2, file_output=var_4, file_path=var_0, output=var_0, color_output=var_5)
    var_7 = module_1.code_to_chars(var_0)
    assert var_7 == '\x1b[Nonem'
    assert module_1.CSI == '\x1b['
    assert module_1.OSC == '\x1b]'
    assert module_1.BEL == '\x07'
    assert f'{type(module_1.Fore).__module__}.{type(module_1.Fore).__qualname__}' == 'colorama.ansi.AnsiFore'
    assert module_1.Fore.BLACK == '\x1b[30m'
    assert module_1.Fore.BLUE == '\x1b[34m'
    assert module_1.Fore.CYAN == '\x1b[36m'
    assert module_1.Fore.GREEN == '\x1b[32m'
    assert module_1.Fore.LIGHTBLACK_EX == '\x1b[90m'
    assert module_1.Fore.LIGHTBLUE_EX == '\x1b[94m'
    assert module_1.Fore.LIGHTCYAN_EX == '\x1b[96m'
    assert module_1.Fore.LIGHTGREEN_EX == '\x1b[92m'
    assert module_1.Fore.LIGHTMAGENTA_EX == '\x1b[95m'
    assert module_1.Fore.LIGHTRED_EX == '\x1b[91m'
    assert module_1.Fore.LIGHTWHITE_EX == '\x1b[97m'
    assert module_1.Fore.LIGHTYELLOW_EX == '\x1b[93m'
    assert module_1.Fore.MAGENTA == '\x1b[35m'
    assert module_1.Fore.RED == '\x1b[31m'
    assert module_1.Fore.RESET == '\x1b[39m'
    assert module_1.Fore.WHITE == '\x1b[37m'
    assert module_1.Fore.YELLOW == '\x1b[33m'
    assert f'{type(module_1.Back).__module__}.{type(module_1.Back).__qualname__}' == 'colorama.ansi.AnsiBack'
    assert module_1.Back.BLACK == '\x1b[40m'
    assert module_1.Back.BLUE == '\x1b[44m'
    assert module_1.Back.CYAN == '\x1b[46m'
    assert module_1.Back.GREEN == '\x1b[42m'
    assert module_1.Back.LIGHTBLACK_EX == '\x1b[100m'
    assert module_1.Back.LIGHTBLUE_EX == '\x1b[104m'
    assert module_1.Back.LIGHTCYAN_EX == '\x1b[106m'
    assert module_1.Back.LIGHTGREEN_EX == '\x1b[102m'
    assert module_1.Back.LIGHTMAGENTA_EX == '\x1b[105m'
    assert module_1.Back.LIGHTRED_EX == '\x1b[101m'
    assert module_1.Back.LIGHTWHITE_EX == '\x1b[107m'
    assert module_1.Back.LIGHTYELLOW_EX == '\x1b[103m'
    assert module_1.Back.MAGENTA == '\x1b[45m'
    assert module_1.Back.RED == '\x1b[41m'
    assert module_1.Back.RESET == '\x1b[49m'
    assert module_1.Back.WHITE == '\x1b[47m'
    assert module_1.Back.YELLOW == '\x1b[43m'
    assert f'{type(module_1.Style).__module__}.{type(module_1.Style).__qualname__}' == 'colorama.ansi.AnsiStyle'
    assert module_1.Style.BRIGHT == '\x1b[1m'
    assert module_1.Style.DIM == '\x1b[2m'
    assert module_1.Style.NORMAL == '\x1b[22m'
    assert module_1.Style.RESET_ALL == '\x1b[0m'
    assert f'{type(module_1.Cursor).__module__}.{type(module_1.Cursor).__qualname__}' == 'colorama.ansi.AnsiCursor'
    var_7.__bool__()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '.U>kq1_\tQl'
    var_1 = ''
    module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = '\nfQf;'
    var_2 = module_0.format_natural(var_1)
    assert var_2 == 'import fQf;'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = 'V[y#\x0cB<P\n'
    var_4 = module_0.format_simplified(var_2)
    assert var_4 == 'fQf;'
    var_5 = True
    var_6 = module_0.show_unified_diff(file_input=var_2, file_output=var_3, file_path=var_0, output=var_0, color_output=var_5)
    var_7 = None
    var_8 = module_1.code_to_chars(var_7)
    assert var_8 == '\x1b[Nonem'
    assert module_1.CSI == '\x1b['
    assert module_1.OSC == '\x1b]'
    assert module_1.BEL == '\x07'
    assert f'{type(module_1.Fore).__module__}.{type(module_1.Fore).__qualname__}' == 'colorama.ansi.AnsiFore'
    assert module_1.Fore.BLACK == '\x1b[30m'
    assert module_1.Fore.BLUE == '\x1b[34m'
    assert module_1.Fore.CYAN == '\x1b[36m'
    assert module_1.Fore.GREEN == '\x1b[32m'
    assert module_1.Fore.LIGHTBLACK_EX == '\x1b[90m'
    assert module_1.Fore.LIGHTBLUE_EX == '\x1b[94m'
    assert module_1.Fore.LIGHTCYAN_EX == '\x1b[96m'
    assert module_1.Fore.LIGHTGREEN_EX == '\x1b[92m'
    assert module_1.Fore.LIGHTMAGENTA_EX == '\x1b[95m'
    assert module_1.Fore.LIGHTRED_EX == '\x1b[91m'
    assert module_1.Fore.LIGHTWHITE_EX == '\x1b[97m'
    assert module_1.Fore.LIGHTYELLOW_EX == '\x1b[93m'
    assert module_1.Fore.MAGENTA == '\x1b[35m'
    assert module_1.Fore.RED == '\x1b[31m'
    assert module_1.Fore.RESET == '\x1b[39m'
    assert module_1.Fore.WHITE == '\x1b[37m'
    assert module_1.Fore.YELLOW == '\x1b[33m'
    assert f'{type(module_1.Back).__module__}.{type(module_1.Back).__qualname__}' == 'colorama.ansi.AnsiBack'
    assert module_1.Back.BLACK == '\x1b[40m'
    assert module_1.Back.BLUE == '\x1b[44m'
    assert module_1.Back.CYAN == '\x1b[46m'
    assert module_1.Back.GREEN == '\x1b[42m'
    assert module_1.Back.LIGHTBLACK_EX == '\x1b[100m'
    assert module_1.Back.LIGHTBLUE_EX == '\x1b[104m'
    assert module_1.Back.LIGHTCYAN_EX == '\x1b[106m'
    assert module_1.Back.LIGHTGREEN_EX == '\x1b[102m'
    assert module_1.Back.LIGHTMAGENTA_EX == '\x1b[105m'
    assert module_1.Back.LIGHTRED_EX == '\x1b[101m'
    assert module_1.Back.LIGHTWHITE_EX == '\x1b[107m'
    assert module_1.Back.LIGHTYELLOW_EX == '\x1b[103m'
    assert module_1.Back.MAGENTA == '\x1b[45m'
    assert module_1.Back.RED == '\x1b[41m'
    assert module_1.Back.RESET == '\x1b[49m'
    assert module_1.Back.WHITE == '\x1b[47m'
    assert module_1.Back.YELLOW == '\x1b[43m'
    assert f'{type(module_1.Style).__module__}.{type(module_1.Style).__qualname__}' == 'colorama.ansi.AnsiStyle'
    assert module_1.Style.BRIGHT == '\x1b[1m'
    assert module_1.Style.DIM == '\x1b[2m'
    assert module_1.Style.NORMAL == '\x1b[22m'
    assert module_1.Style.RESET_ALL == '\x1b[0m'
    assert f'{type(module_1.Cursor).__module__}.{type(module_1.Cursor).__qualname__}' == 'colorama.ansi.AnsiCursor'
    var_8.__bool__()