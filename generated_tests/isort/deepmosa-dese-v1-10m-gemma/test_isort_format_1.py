# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.format as module_0
import colorama.initialise as module_1
import re as module_2

def test_case_0():
    var_0 = 'd=&@z(rI'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == 'd=&@z(rI'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

def test_case_1():
    var_0 = '+UT'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import +UT'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = False
    module_0.ask_whether_to_apply_changes_to_file(var_0)

def test_case_4():
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
def test_case_5():
    var_0 = 'ImqC{T|"\tctam*029N'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import ImqC{T|"\tctam*029N'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = module_0.format_simplified(var_1)
    assert var_2 == 'ImqC{T|"\tctam*029N'
    var_3 = None
    var_4 = module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_3, color_output=var_0)
    module_0.ask_whether_to_apply_changes_to_file(var_3)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.remove_whitespace(var_0)

def test_case_7():
    pass

def test_case_8():
    var_0 = '|A#|R[h?#C-dAUB]9'
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == '|A#|R[h?#C-dAUB]9'
    assert var_2.error_message == '|A#|R[h?#C-dAUB]9'
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = '0E"ij"'
    var_4 = module_0.format_simplified(var_3)
    assert var_4 == '0E"ij"'
    var_5 = var_2.error(var_0)
    var_6 = ''
    var_7 = var_2.error(var_6)
    var_8 = None
    var_9 = module_0.ColoramaPrinter(var_0, var_0, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_9.output).__module__}.{type(var_9.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_9.success_message == '|A#|R[h?#C-dAUB]9'
    assert var_9.error_message == '|A#|R[h?#C-dAUB]9'
    assert var_9.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_9.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_9.ADDED_LINE == '\x1b[32m'
    assert var_9.REMOVED_LINE == '\x1b[31m'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'Imq6{T|"\tctam*029N'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import Imq6{T|"\tctam*029N'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0, output=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_2 = 'o#a0&;4zB5'
    var_3 = var_1.success(var_2)
    var_4 = module_1.just_fix_windows_console()
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
    var_5 = '( \t=7"R'
    module_0.remove_whitespace(var_0, var_5)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'tg+::A4x'
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message is None
    assert var_2.error_message == 'tg+::A4x'
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = None
    var_4 = module_0.create_terminal_printer(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == ''
    assert var_4.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5 = None
    var_6 = var_2.diff_line(var_0)
    var_7 = var_4.diff_line(var_0)
    var_8 = ''
    var_9 = module_2.template(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 're.Pattern'
    assert module_2.ASCII == module_2.RegexFlag.ASCII
    assert module_2.A == module_2.RegexFlag.ASCII
    assert module_2.IGNORECASE == module_2.RegexFlag.IGNORECASE
    assert module_2.I == module_2.RegexFlag.IGNORECASE
    assert module_2.LOCALE == module_2.RegexFlag.LOCALE
    assert module_2.L == module_2.RegexFlag.LOCALE
    assert module_2.UNICODE == module_2.RegexFlag.UNICODE
    assert module_2.U == module_2.RegexFlag.UNICODE
    assert module_2.MULTILINE == module_2.RegexFlag.MULTILINE
    assert module_2.M == module_2.RegexFlag.MULTILINE
    assert module_2.DOTALL == module_2.RegexFlag.DOTALL
    assert module_2.S == module_2.RegexFlag.DOTALL
    assert module_2.VERBOSE == module_2.RegexFlag.VERBOSE
    assert module_2.X == module_2.RegexFlag.VERBOSE
    assert module_2.TEMPLATE == module_2.RegexFlag.TEMPLATE
    assert module_2.T == module_2.RegexFlag.TEMPLATE
    assert module_2.DEBUG == module_2.RegexFlag.DEBUG
    assert f'{type(module_2.Pattern.pattern).__module__}.{type(module_2.Pattern.pattern).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Pattern.flags).__module__}.{type(module_2.Pattern.flags).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Pattern.groups).__module__}.{type(module_2.Pattern.groups).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_2.Pattern.groupindex).__module__}.{type(module_2.Pattern.groupindex).__qualname__}' == 'builtins.getset_descriptor'
    var_9.error(var_5)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '1]Ap}v3SaPi.xH'
    var_1 = None
    var_2 = module_0.create_terminal_printer(var_1, error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message is None
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_3 = module_0.format_natural(var_0)
    assert var_3 == 'from 1]Ap}v3SaPi import xH'
    var_4 = module_0.format_simplified(var_3)
    assert var_4 == '1]Ap}v3SaPi.xH'
    var_5 = module_0.show_unified_diff(file_input=var_0, file_output=var_3, file_path=var_1, color_output=var_0)
    module_0.ask_whether_to_apply_changes_to_file(var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'ImqC{T|"\tctam*029N'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == 'ImqC{T|"\tctam*029N'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = None
    var_3 = module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_2, color_output=var_0)
    module_0.ask_whether_to_apply_changes_to_file(var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'ImqC{T|"\tctam*029N'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import ImqC{T|"\tctam*029N'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = module_0.format_simplified(var_1)
    assert var_2 == 'ImqC{T|"\tctam*029N'
    var_3 = None
    var_4 = 'h&vPT|zy\r>'
    var_5 = module_0.BasicPrinter(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.BasicPrinter'
    assert var_5.output == 'h&vPT|zy\r>'
    assert var_5.success_message is None
    assert var_5.error_message is None
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_6 = module_0.show_unified_diff(file_input=var_4, file_output=var_4, file_path=var_3)
    module_0.ask_whether_to_apply_changes_to_file(var_3)

def test_case_15():
    var_0 = 'ImqC{T|"\tctam*029N'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import ImqC{T|"\tctam*029N'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = module_0.format_simplified(var_1)
    assert var_2 == 'ImqC{T|"\tctam*029N'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'ImqC{T|"\tctam*029N'
    var_1 = None
    var_2 = module_0.create_terminal_printer(var_1, error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == ''
    assert var_2.error_message is None
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_3 = module_0.format_natural(var_0)
    assert var_3 == 'import ImqC{T|"\tctam*029N'
    var_4 = True
    var_5 = module_0.create_terminal_printer(var_4, error=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_5.output).__module__}.{type(var_5.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_5.success_message == ''
    assert var_5.error_message is None
    assert var_5.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_5.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_5.ADDED_LINE == '\x1b[32m'
    assert var_5.REMOVED_LINE == '\x1b[31m'
    var_6 = module_0.format_natural(var_3)
    assert var_6 == 'import ImqC{T|"\tctam*029N'
    var_7 = '45&~yF\rXe/Eq-'
    var_8 = module_0.format_simplified(var_7)
    assert var_8 == '45&~yF\rXe/Eq-'
    var_9 = None
    module_0.show_unified_diff(file_input=var_7, file_output=var_9, file_path=var_1, color_output=var_1)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'he#/,T_hjC.]6'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'from he#/,T_hjC import ]6'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = module_0.format_natural(var_1)
    assert var_2 == 'from he#/,T_hjC import ]6'
    var_3 = '{x=\t)Zez'
    module_0.ask_whether_to_apply_changes_to_file(var_3)