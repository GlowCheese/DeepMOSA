# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.format as module_0
import colorama.ansi as module_1
import colorama.initialise as module_2
import colorama.winterm as module_3

def test_case_0():
    var_0 = '0 (#'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == '0 (#'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

def test_case_1():
    var_0 = 'i7'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import i7'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'lsQ/i-\x0bn<[$xl}^'
    var_1 = None
    var_2 = module_1.clear_line()
    assert var_2 == '\x1b[2K'
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
    var_3 = module_0.format_simplified(var_0)
    assert var_3 == 'lsQ/i-\x0bn<[$xl}^'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = None
    var_5 = '48\nCMe$Y7={kY`t'
    var_6 = "X'\x0ctbD_NzkG\x0b"
    var_7 = True
    var_8 = module_0.show_unified_diff(file_input=var_5, file_output=var_6, file_path=var_1, color_output=var_7)
    var_2.__setattr__(var_1, var_4, var_2)

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
    var_3 = module_2.just_fix_windows_console()
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
    var_3.__exit__()

def test_case_6():
    var_0 = None
    var_1 = module_0.ColoramaPrinter(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message is None
    assert var_1.error_message is None
    assert var_1.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_1.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_1.ADDED_LINE == '\x1b[32m'
    assert var_1.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

def test_case_7():
    var_0 = False
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

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.remove_whitespace(var_0)

def test_case_10():
    pass

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'an8/8]uPf+v-/5<a'
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
    var_3 = module_0.format_natural(var_0)
    assert var_3 == 'import an8/8]uPf+v-/5<a'
    var_2.success(var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
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
    var_2 = var_1.error(var_0)
    var_3 = module_1.clear_line()
    assert var_3 == '\x1b[2K'
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
    var_4 = module_0.format_simplified(var_3)
    assert var_4 == '\x1b[2K'
    var_5 = "X'\x0ctbD_NzkG\x0b"
    var_6 = True
    var_7 = module_0.format_natural(var_5)
    assert var_7 == "import X'\x0ctbD_NzkG"
    var_8 = module_0.show_unified_diff(file_input=var_3, file_output=var_5, file_path=var_0, color_output=var_6)
    var_9 = module_3.enable_vt_processing(var_6)
    assert var_9 is False
    var_10 = module_2.init(var_0)
    assert f'{type(module_2.orig_stdout).__module__}.{type(module_2.orig_stdout).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.orig_stdout.mode == 'w'
    assert f'{type(module_2.orig_stderr).__module__}.{type(module_2.orig_stderr).__qualname__}' == '_io.TextIOWrapper'
    assert module_2.orig_stderr.mode == 'w'
    assert f'{type(module_2.wrapped_stdout).__module__}.{type(module_2.wrapped_stdout).__qualname__}' == 'colorama.ansitowin32.StreamWrapper'
    assert f'{type(module_2.wrapped_stderr).__module__}.{type(module_2.wrapped_stderr).__qualname__}' == 'colorama.ansitowin32.StreamWrapper'
    assert module_2.atexit_done is True
    assert module_2.fixed_windows_console is False
    module_0.ask_whether_to_apply_changes_to_file(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
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

def test_case_14():
    var_0 = 'i7.'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'from i7 import '
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_1.clear_line()
    assert var_0 == '\x1b[2K'
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
    var_1 = '48\nCMe$Y7={kY`t'
    var_2 = "X'\x0ctbD_NzkG\x0b"
    var_3 = True
    module_0.show_unified_diff(file_input=var_1, file_output=var_2, file_path=var_0, color_output=var_3)

def test_case_16():
    var_0 = 'Y\n'
    var_1 = False
    var_2 = module_0.create_terminal_printer(var_1, error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message == 'Y\n'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_3 = '"M'
    var_4 = module_0.format_natural(var_3)
    assert var_4 == 'import "M'
    var_5 = '{[g<C%p5IeW'
    var_6 = module_0.BasicPrinter(var_4, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message == '{[g<C%p5IeW'
    assert var_6.error_message == 'import "M'
    var_7 = module_0.format_simplified(var_4)
    assert var_7 == '"M'

def test_case_17():
    var_0 = 'from os import path'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'from os import path'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'lsQ/<zi-\x0bn<[$xl}^'
    var_1 = None
    var_2 = module_0.create_terminal_printer(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message == ''
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_3 = module_1.clear_line()
    assert var_3 == '\x1b[2K'
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
    var_4 = module_0.format_simplified(var_0)
    assert var_4 == 'lsQ/<zi-\x0bn<[$xl}^'
    var_5 = module_0.format_natural(var_0)
    assert var_5 == 'import lsQ/<zi-\x0bn<[$xl}^'
    var_6 = module_0.format_natural(var_5)
    assert var_6 == 'import lsQ/<zi-\x0bn<[$xl}^'
    var_3.write(var_3)

def test_case_19():
    var_0 = 'from os import path'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == 'os.path'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'