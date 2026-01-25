# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.format as module_0
import re as module_1

def test_case_0():
    var_0 = '5T@GT%H.4'
    var_1 = module_0.format_simplified(var_0)
    assert var_1 == '5T@GT%H.4'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

def test_case_1():
    var_0 = '/`unVyJHa}'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import /`unVyJHa}'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_0, output=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'D8==c3ZT3.'
    var_1 = '=T8d]7*6BMPc<\nQt/[<'
    var_2 = module_0.BasicPrinter(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == '=T8d]7*6BMPc<\nQt/[<'
    assert var_2.error_message == '=T8d]7*6BMPc<\nQt/[<'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_3 = var_2.error(var_1)
    var_4 = None
    var_5 = module_0.show_unified_diff(file_input=var_0, file_output=var_1, file_path=var_4)
    module_0.remove_whitespace(var_4, var_4)

def test_case_4():
    var_0 = '?=3='
    var_1 = None
    var_2 = '0o3\ng/p^0-H#'
    var_3 = module_0.show_unified_diff(file_input=var_0, file_output=var_0, file_path=var_1)
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_4 = module_0.format_natural(var_2)
    assert var_4 == 'import 0o3\ng/p^0-H#'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'iHk%_'
    var_1 = module_0.create_terminal_printer(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert var_1.output == 'iHk%_'
    assert var_1.success_message == ''
    assert var_1.error_message == ''
    assert var_1.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_1.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_1.ADDED_LINE == '\x1b[32m'
    assert var_1.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = module_0.format_natural(var_0)
    assert var_2 == 'import iHk%_'
    var_3 = True
    var_4 = '%;kKLANQ\x0cH]/6=?G'
    var_5 = module_0.create_terminal_printer(var_3, success=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_5.output).__module__}.{type(var_5.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_5.success_message == '%;kKLANQ\x0cH]/6=?G'
    assert var_5.error_message == ''
    assert var_5.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_5.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_5.ADDED_LINE == '\x1b[32m'
    assert var_5.REMOVED_LINE == '\x1b[31m'
    var_6 = None
    module_1.match(var_6, var_6)

def test_case_6():
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

def test_case_7():
    var_0 = 2471.8585
    var_1 = 'Svm\x0b*H,z3M^8UVe\\\x0b<'
    var_2 = module_0.create_terminal_printer(var_0, error=var_1, success=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == 'Svm\x0b*H,z3M^8UVe\\\x0b<'
    assert var_2.error_message == 'Svm\x0b*H,z3M^8UVe\\\x0b<'
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

def test_case_8():
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
    var_3 = var_1.error(var_2)
    var_4 = var_1.error_message
    assert var_4 == ''
    var_5 = var_1.success_message
    assert var_5 == ''
    var_6 = var_1.output
    module_0.format_simplified(var_6)

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
    var_0 = 'M;cOAMnn7H=R\r'
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == 'M;cOAMnn7H=R\r'
    assert var_2.error_message == 'M;cOAMnn7H=R\r'
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = var_2.diff_line(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = '%2a`\x0bpfzruR0J'
    var_1 = module_0.remove_whitespace(var_0)
    assert var_1 == '%2a`\x0bpfzruR0J'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_2 = '\r^'
    var_3 = module_0.create_terminal_printer(var_2, error=var_1, success=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_3.output).__module__}.{type(var_3.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_3.success_message == '\r^'
    assert var_3.error_message == '%2a`\x0bpfzruR0J'
    assert var_3.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_3.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_3.ADDED_LINE == '\x1b[32m'
    assert var_3.REMOVED_LINE == '\x1b[31m'
    var_4 = '|5x_o:xi.14$][&.'
    var_5 = module_0.format_natural(var_4)
    assert var_5 == 'from |5x_o:xi.14$][& import '
    var_6 = module_0.remove_whitespace(var_1, var_2)
    assert var_6 == '%2a`\x0bpfzruR0J'
    var_7 = None
    var_8 = 'jtU3K4NW]B*}'
    module_0.show_unified_diff(file_input=var_8, file_output=var_7, file_path=var_7, output=var_1)

@pytest.mark.xfail(strict=True)
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
    var_2 = 'z_V?V/\x0bHP6E\x0b@a'
    var_3 = 'fM~b3>GO1|]P\t\n\tCk\n0'
    var_4 = False
    module_0.show_unified_diff(file_input=var_2, file_output=var_3, file_path=var_2, color_output=var_4)

def test_case_17():
    var_0 = 'import os'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'import os'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '?=3='
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == '?=3='
    assert var_2.error_message == '?=3='
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = '=T8d]7*6BMPc<\nQt/[<'
    var_4 = module_0.BasicPrinter(var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == '=T8d]7*6BMPc<\nQt/[<'
    assert var_4.error_message == '=T8d]7*6BMPc<\nQt/[<'
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    var_5 = var_4.error(var_3)
    var_6 = None
    var_7 = module_0.show_unified_diff(file_input=var_3, file_output=var_3, file_path=var_6)
    var_8 = module_0.format_natural(var_3)
    assert var_8 == 'import =T8d]7*6BMPc<\nQt/[<'
    var_9 = "{.]QWX`#&'.l?|~h"
    var_10 = '4"2d'
    var_11 = module_0.format_simplified(var_10)
    assert var_11 == '4"2d'
    var_12 = module_0.ColoramaPrinter(var_9, var_6, var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_12.output).__module__}.{type(var_12.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_12.success_message is None
    assert var_12.error_message == "{.]QWX`#&'.l?|~h"
    assert var_12.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_12.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_12.ADDED_LINE == '\x1b[32m'
    assert var_12.REMOVED_LINE == '\x1b[31m'
    var_13 = '-KdkZ 0E!3|kZ'
    var_14 = var_12.diff_line(var_13)
    module_0.show_unified_diff(file_input=var_1, file_output=var_0, file_path=var_5, output=var_1, color_output=var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = '?=b='
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == '?=b='
    assert var_2.error_message == '?=b='
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = '=T8d]7*6BMPc<\nQt/[<'
    var_4 = var_2.error(var_3)
    var_5 = None
    var_6 = module_0.show_unified_diff(file_input=var_3, file_output=var_3, file_path=var_5)
    var_7 = module_0.format_natural(var_3)
    assert var_7 == 'import =T8d]7*6BMPc<\nQt/[<'
    var_8 = '4"2d'
    var_9 = var_2.style_text(var_1)
    var_10 = module_0.format_simplified(var_8)
    assert var_10 == '4"2d'
    var_11 = module_0.format_simplified(var_7)
    assert var_11 == '=T8d]7*6BMPc<\nQt/[<'
    var_12 = module_0.ColoramaPrinter(var_1, var_1, var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_12.output).__module__}.{type(var_12.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_12.success_message is None
    assert var_12.error_message is None
    assert var_12.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_12.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_12.ADDED_LINE == '\x1b[32m'
    assert var_12.REMOVED_LINE == '\x1b[31m'
    var_13 = ''
    module_0.remove_whitespace(var_9, var_13)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = '?=b='
    var_1 = None
    var_2 = module_0.ColoramaPrinter(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_2.output).__module__}.{type(var_2.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_2.success_message == '?=b='
    assert var_2.error_message == '?=b='
    assert var_2.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_2.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_2.ADDED_LINE == '\x1b[32m'
    assert var_2.REMOVED_LINE == '\x1b[31m'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    var_3 = '=T8d]7*6BMPc<\nQt/[<'
    var_4 = var_2.error(var_3)
    var_5 = module_0.show_unified_diff(file_input=var_3, file_output=var_3, file_path=var_1)
    var_6 = '=T8d]7*6BMPc<\nQt/[<'
    var_7 = var_2.error(var_6)
    var_8 = 'R'
    var_9 = 'etmgatS'
    var_10 = True
    var_11 = module_0.show_unified_diff(file_input=var_8, file_output=var_9, file_path=var_1, output=var_5, color_output=var_10)
    module_0.format_natural(var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = b'\x84d'
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
    var_2 = '/\\Et'
    var_3 = None
    var_4 = module_0.ColoramaPrinter(var_2, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'isort.format.ColoramaPrinter'
    assert f'{type(var_4.output).__module__}.{type(var_4.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_4.success_message == '/\\Et'
    assert var_4.error_message == '/\\Et'
    assert var_4.ERROR == '\x1b[31mERROR\x1b[0m'
    assert var_4.SUCCESS == '\x1b[32mSUCCESS\x1b[0m'
    assert var_4.ADDED_LINE == '\x1b[32m'
    assert var_4.REMOVED_LINE == '\x1b[31m'
    var_5 = '=T8d]7*6BMPc<\nQt/[<'
    var_6 = var_4.error(var_3)
    var_7 = False
    var_8 = module_0.show_unified_diff(file_input=var_2, file_output=var_5, file_path=var_6, color_output=var_7)
    var_9 = 'Czfm:v{<v7itdJ.o\\'
    var_10 = module_0.format_natural(var_9)
    assert var_10 == 'from Czfm:v{<v7itdJ import o\\'
    var_11 = var_4.style_text(var_6)
    var_12 = module_0.format_simplified(var_10)
    assert var_12 == 'Czfm:v{<v7itdJ.o\\'
    var_13 = module_0.create_terminal_printer(var_3, success=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'isort.format.BasicPrinter'
    assert f'{type(var_13.output).__module__}.{type(var_13.output).__qualname__}' == '_io.TextIOWrapper'
    assert var_13.success_message is None
    assert var_13.error_message == ''
    assert module_0.BasicPrinter.ERROR == 'ERROR'
    assert module_0.BasicPrinter.SUCCESS == 'SUCCESS'
    module_1.escape(var_3)

def test_case_22():
    var_0 = 'from os import path'
    var_1 = module_0.format_natural(var_0)
    assert var_1 == 'from os import path'
    assert module_0.colorama_unavailable is False
    assert f'{type(module_0.ADDED_LINE_PATTERN).__module__}.{type(module_0.ADDED_LINE_PATTERN).__qualname__}' == 're.Pattern'
    assert f'{type(module_0.REMOVED_LINE_PATTERN).__module__}.{type(module_0.REMOVED_LINE_PATTERN).__qualname__}' == 're.Pattern'