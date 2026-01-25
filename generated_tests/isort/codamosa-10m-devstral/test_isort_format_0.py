# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.format as module_0
import re as module_1
import colorama.initialise as module_2
import colorama.ansi as module_3

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = "}$iYpD'ar\x0b}\x0c94xG5jJ"
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == "}$iYpD'ar\x0b}\x0c94xG5jJ"
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = None
    module_1.finditer(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'jGa-[iID@`[Og/{aD*M'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import jGa-[iID@`[Og/{aD*M'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = '%;kKLANQ\x0cH]/6=?G'
    var_3 = module_0.create_terminal_printer(var_2, success=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_3.output).__module__}.{type(var_3.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_3.success_message == '%;kKLANQ\x0cH]/6=?G'
    assert var_3.error_message == ''
    assert var_3.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_3.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_3.ADDED_LINE == '\x1b[32m'
    assert var_3.REMOVED_LINE == '\x1b[31m'
    var_4 = None
    module_1.match(var_4, var_4)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = '\x0bO,g{zlC]Up#'
    module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_0)

def test_case_3():
    var_0 = 'D8==c3ZT3.'
    var_1 = '=T8d]7*6BMPc<\nQt/[<'
    var_2 = None
    var_3 = module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_2)
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = '+G;CRGkdW5"s0>s_tX'
    var_5 = module_0.format_natural(var_0)
    assert var_5 == 'from D8==c3ZT3 import '
    var_6 = module_0.create_terminal_printer(var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message == ''
    assert var_6.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_7 = module_0.format_natural(var_4)
    assert var_7 == 'import +G;CRGkdW5"s0>s_tX'
    var_8 = 'K\x0bZ\\LC:>j\\0s`'
    var_9 = module_0.format_simplified(var_8)
    assert var_9 == 'K\x0bZ\\LC:>j\\0s`'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = "z'E\x0bU\\fyj"
    var_1 = None
    var_2 = module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_1, output=var_1)
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = None
    var_4 = None
    var_5 = module_2.init(strip=var_4, wrap=var_4)
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
    var_3.__getitem__(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = True
    var_1 = b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    var_2 = module_0.create_terminal_printer(var_0, var_1, success=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert var_2.output == b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    assert var_2.success_message == b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = 'D8==c3ZT3.'
    var_4 = module_0.format_simplified(var_3)
    assert var_4 == 'D8==c3ZT3.'
    var_5 = '=T8d]7*6BMPc<\nQt/[<'
    var_6 = module_0.BasicPrinter(var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message == '=T8d]7*6BMPc<\nQt/[<'
    assert var_6.error_message == '=T8d]7*6BMPc<\nQt/[<'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_7 = var_6.error(var_5)
    var_8 = None
    var_9 = module_0.show_unified_diff(file_input=var_3, file_output=var_5, file_path=var_8)
    var_10 = var_6.error(var_9)
    var_11 = module_0.BasicPrinter(var_5, var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_11.output).__module__}.{type(var_11.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_11.success_message == 'D8==c3ZT3.'
    assert var_11.error_message == '=T8d]7*6BMPc<\nQt/[<'
    module_1.template(var_6)

def test_case_6():
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

def test_case_7():
    var_0 = None
    var_1 = 'mk3!$e9u}U.-FRp\t]'
    var_2 = module_0.ColoramaPrinter(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == 'mk3!$e9u}U.-FRp\t]'
    assert var_2.error_message is None
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

def test_case_8():
    var_0 = True
    var_1 = ':A'
    var_2 = module_0.create_terminal_printer(var_0, success=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ':A'
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.remove_whitespace(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.format_simplified(var_0)

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
    var_0 = 'D8==c3ZT3.'
    var_1 = module_0.BasicPrinter(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message == 'D8==c3ZT3.'
    assert var_1.error_message == 'D8==c3ZT3.'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_2 = None
    var_3 = 'ZB7\x0b\\=a:\rxmk$J'
    var_4 = module_0.show_unified_diff(file_input=var_0, file_output=var_3, file_path=var_2)
    var_5 = module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_2)
    var_6 = module_0.BasicPrinter(var_5, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message is None
    assert var_6.error_message is None
    var_7 = '+G;CRGkdW5"s0>s_tX'
    var_8 = var_1.error(var_7)
    var_9 = 'M@eH'
    var_10 = module_0.ColoramaPrinter(var_5, var_5, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_10.output).__module__}.{type(var_10.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_10.success_message is None
    assert var_10.error_message is None
    assert var_10.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_10.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_10.ADDED_LINE == '\x1b[32m'
    assert var_10.REMOVED_LINE == '\x1b[31m'
    var_11 = False
    var_12 = module_0.create_terminal_printer(var_11, var_5, success=var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_12.output).__module__}.{type(var_12.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_12.success_message is None
    assert var_12.error_message == ''
    var_13 = module_0.format_natural(var_9)
    assert var_13 == 'import M@eH'
    var_14 = var_10.diff_line(var_0)
    module_0.format_natural(var_6)

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
    var_0 = None
    var_1 = 'mk3!$e9u}U.-FRp\t]'
    var_2 = module_0.ColoramaPrinter(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == 'mk3!$e9u}U.-FRp\t]'
    assert var_2.error_message is None
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = '=>/$@}H(aK[h9!O2\\0?'
    var_4 = var_2.diff_line(var_3)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = '=T8d]7*6BMPc<\nQt/[<'
    var_1 = module_0.BasicPrinter(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_1.output).__module__}.{type(var_1.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_1.success_message == '=T8d]7*6BMPc<\nQt/[<'
    assert var_1.error_message == '=T8d]7*6BMPc<\nQt/[<'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_2 = var_1.error(var_0)
    var_3 = var_1.error(var_2)
    var_4 = ':Hg)G^\x0b5~nsb\rh^OEp'
    module_0.show_unified_diff(file_input=var_2, file_output=var_4, file_path=var_0)

def test_case_16():
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
    var_2 = module_3.clear_line()
    assert var_2 == '\x1b[2K'
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
    var_3 = var_1.style_text(var_2)
    assert var_3 == '\x1b[2K'

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'F|W'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == 'F|W'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = 'D8==c3ZT3.'
    var_3 = 'I1@[\tS;T'
    var_4 = module_0.BasicPrinter(var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == 'I1@[\tS;T'
    assert var_4.error_message == 'F|W'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5 = None
    var_6 = 'AyGPNd'
    var_7 = 'Ie\\&+T58)>DKF'
    var_8 = module_0.ColoramaPrinter(var_2, var_7, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_8.output).__module__}.{type(var_8.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_8.success_message == 'Ie\\&+T58)>DKF'
    assert var_8.error_message == 'D8==c3ZT3.'
    assert var_8.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_8.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_8.ADDED_LINE == '\x1b[32m'
    assert var_8.REMOVED_LINE == '\x1b[31m'
    var_9 = var_8.error(var_5)
    var_10 = '\x0bB_WJprF#9\tektZ|1 '
    var_11 = '\\l0Xa=JPI)\\&)s'
    var_12 = True
    var_13 = module_0.show_unified_diff(file_input=var_10, file_output=var_11, file_path=var_9, color_output=var_12)
    var_14 = '`Je{>>A6'
    var_15 = module_0.BasicPrinter(var_6, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_15.output).__module__}.{type(var_15.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_15.success_message == '`Je{>>A6'
    assert var_15.error_message == 'AyGPNd'
    var_16 = module_0.ColoramaPrinter(var_13, var_9, var_5)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_16.output).__module__}.{type(var_16.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_16.success_message is None
    assert var_16.error_message is None
    assert var_16.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_16.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_16.ADDED_LINE == '\x1b[32m'
    assert var_16.REMOVED_LINE == '\x1b[31m'
    module_0.format_natural(var_16)

def test_case_18():
    var_0 = 'D8==c3ZT3.'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == 'D8==c3ZT3.'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = '=T8d]7*6BMPc<\nQt/[<'
    var_3 = module_0.format_natural(var_2)
    assert var_3 == 'import =T8d]7*6BMPc<\nQt/[<'
    var_4 = module_0.BasicPrinter(var_2, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == 'D8==c3ZT3.'
    assert var_4.error_message == '=T8d]7*6BMPc<\nQt/[<'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5 = '+G;CRGkdW5"s0>s_tX'
    var_6 = var_4.error(var_2)
    var_7 = module_0.format_natural(var_5)
    assert var_7 == 'import +G;CRGkdW5"s0>s_tX'
    var_8 = module_0.BasicPrinter(var_2, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_8.output).__module__}.{type(var_8.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_8.success_message is None
    assert var_8.error_message == '=T8d]7*6BMPc<\nQt/[<'
    var_9 = module_0.format_natural(var_3)
    assert var_9 == 'import =T8d]7*6BMPc<\nQt/[<'
    var_10 = 'Q2BAl<XP0rwSc[9"v'
    var_11 = module_0.BasicPrinter(var_6, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_11.output).__module__}.{type(var_11.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_11.success_message == 'Q2BAl<XP0rwSc[9"v'
    assert var_11.error_message is None
    var_12 = 'V88]i.4'
    var_13 = module_0.format_simplified(var_12)
    assert var_13 == 'V88]i.4'

def test_case_19():
    var_0 = 'D8==c3ZT3.'
    var_1 = '=T8d]7*6BMPc<\nQt/[<'
    var_2 = None
    var_3 = module_0.BasicPrinter(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_3.output).__module__}.{type(var_3.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_3.success_message == 'D8==c3ZT3.'
    assert var_3.error_message is None
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_4 = module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_2)
    var_5 = None
    var_6 = '+G;CRGkdW5"s0>s_tX'
    var_7 = module_0.format_natural(var_0)
    assert var_7 == 'from D8==c3ZT3 import '
    var_8 = None
    var_9 = module_0.create_terminal_printer(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_9.output).__module__}.{type(var_9.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_9.success_message == ''
    assert var_9.error_message == ''
    var_10 = module_0.format_natural(var_6)
    assert var_10 == 'import +G;CRGkdW5"s0>s_tX'
    var_11 = module_0.BasicPrinter(var_5, var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_11.output).__module__}.{type(var_11.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_11.success_message is None
    assert var_11.error_message is None
    var_12 = module_0.format_simplified(var_10)
    assert var_12 == '+G;CRGkdW5"s0>s_tX'

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = True
    var_1 = b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    var_2 = module_0.create_terminal_printer(var_0, var_1, success=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert var_2.output == b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    assert var_2.success_message == b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = 'D8==c3ZT3.'
    var_4 = module_0.format_simplified(var_3)
    assert var_4 == 'D8==c3ZT3.'
    var_5 = module_0.format_natural(var_3)
    assert var_5 == 'from D8==c3ZT3 import '
    var_6 = module_0.BasicPrinter(var_3, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_6.output).__module__}.{type(var_6.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_6.success_message == 'D8==c3ZT3.'
    assert var_6.error_message == 'D8==c3ZT3.'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_7 = module_0.format_simplified(var_5)
    assert var_7 == 'D8==c3ZT3 import'
    var_8 = None
    var_9 = module_0.show_unified_diff(file_input=var_3, file_output=var_3, file_path=var_8)
    var_10 = '+G;CRGkdW5"s0>s_tX'
    var_11 = var_6.error(var_10)
    var_6.diff_line(var_8)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = True
    var_1 = b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    var_2 = module_0.create_terminal_printer(var_0, var_1, success=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert var_2.output == b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    assert var_2.success_message == b'\xc3\\6\x02$\x0c\xa375\x10\xadVM\x1e;\x90\xce3\x83'
    assert var_2.error_message == ''
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = 'D8==c3ZT3.'
    var_4 = module_0.format_simplified(var_3)
    assert var_4 == 'D8==c3ZT3.'
    var_5 = module_0.format_natural(var_4)
    assert var_5 == 'from D8==c3ZT3 import '
    var_6 = None
    var_7 = '=T8d]7*6BMPc<\nQt/[<'
    var_8 = module_0.format_natural(var_5)
    assert var_8 == 'from D8==c3ZT3 import'
    var_9 = 'cj[`p=p6aFN[#'
    var_10 = module_0.BasicPrinter(var_4, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_10.output).__module__}.{type(var_10.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_10.success_message == 'cj[`p=p6aFN[#'
    assert var_10.error_message == 'D8==c3ZT3.'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_11 = var_10.error(var_7)
    var_12 = '\x0bzWm%Gx@R ^Fw\x0b\t+(F'
    module_0.show_unified_diff(file_input=var_12, file_output=var_6, file_path=var_11, output=var_6)