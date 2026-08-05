# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import isort.sorting as module_0
import builtins as module_1
import re as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.sort(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = '2T\x0b'
    var_1 = None
    module_0.module_key(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'xEz{` rUe'
    var_1 = None
    var_2 = True
    module_0.module_key(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.naturally(var_0, reverse=var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = '\\r\\Td"9"jZ;'
    var_1 = 'n3}B/'
    var_2 = [var_0, var_0, var_1]
    var_3 = True
    var_4 = module_0.naturally(var_2, reverse=var_3)
    assert module_0.TYPE_CHECKING is False
    var_5 = True
    var_6 = None
    var_7 = None
    module_0.module_key(var_6, var_6, var_7, var_6, var_5)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = '19;)NYiao%yu'
    var_1 = None
    var_2 = True
    module_0.module_key(var_0, var_1, ignore_case=var_2, section_name=var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'order_by_type'
    var_3 = 'constants'
    var_4 = 'classes'
    var_5 = 'variables'
    var_6 = 'case_sensitive'
    var_7 = 'length_sort'
    var_8 = 'length_sort_straight'
    var_9 = 'length_sort_sections'
    var_10 = 'force_to_top'
    var_11 = False
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = True
    var_16 = []
    var_17 = []
    var_18 = {var_3: var_11, var_2: var_11, var_3: var_12, var_4: var_13, var_5: var_14, var_6: var_15, var_7: var_11, var_8: var_11, var_9: var_16, var_10: var_17}
    var_19 = [var_0, var_1, var_18]
    var_20 = {}
    var_21 = module_1.type(*var_19, **var_20)
    var_22 = var_21()
    module_0.naturally(var_5, var_22)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'order_by_type'
    var_3 = 'constants'
    var_4 = 'variables'
    var_5 = 'case_sensitive'
    var_6 = 'length_sort'
    var_7 = 'length_sort_straight'
    var_8 = 'length_sort_sections'
    var_9 = False
    var_10 = []
    var_11 = []
    var_12 = True
    var_13 = []
    var_14 = []
    var_15 = {var_7: var_9, var_2: var_9, var_3: var_11, var_5: var_10, var_4: var_11, var_5: var_12, var_6: var_9, var_7: var_9, var_8: var_13, var_7: var_14}
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_1.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = 'some_module'
    module_0.module_key(var_20, var_19, var_12)

def test_case_8():
    var_0 = ()
    var_1 = 'order_by_type'
    var_2 = 'constants'
    var_3 = 'classes'
    var_4 = 'variables'
    var_5 = 'case_sensitive'
    var_6 = 'length_sort'
    var_7 = 'length_sort_sections'
    var_8 = 'force_to_top'
    var_9 = []
    var_10 = []
    var_11 = True
    var_12 = []
    var_13 = {var_2: var_11, var_1: var_11, var_2: var_8, var_3: var_9, var_4: var_10, var_5: var_11, var_6: var_11, var_8: var_11, var_7: var_8, var_8: var_12}
    var_14 = [var_2, var_0, var_13]
    var_15 = {}
    var_16 = module_1.type(*var_14, **var_15)
    var_17 = var_16()
    var_18 = module_0.module_key(var_1, var_17, var_11)
    assert var_18 == 'BC13:order_by_type'
    assert module_0.TYPE_CHECKING is False

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ()
    var_1 = 'reverse_relative'
    var_2 = 'order_by_type'
    var_3 = 'constants'
    var_4 = 'classes'
    var_5 = 'case_sensitive'
    var_6 = 'length_sort'
    var_7 = 'length_sort_sections'
    var_8 = 'force_to_top'
    var_9 = False
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = True
    var_14 = []
    var_15 = module_2.purge()
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
    var_16 = {var_1: var_9, var_2: var_9, var_3: var_10, var_4: var_11, var_6: var_12, var_5: var_13, var_6: var_9, var_8: var_9, var_7: var_14, var_8: var_15}
    var_17 = [var_1, var_0, var_16]
    var_18 = {}
    var_19 = module_1.type(*var_17, **var_18)
    var_20 = var_19()
    module_0.module_key(var_1, var_20, var_13)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '.Nxa\\'
    var_1 = None
    var_2 = True
    module_0.module_key(var_0, var_1, ignore_case=var_2, section_name=var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort_straight'
    var_9 = 'length_sort_sections'
    var_10 = 'force_to_top'
    var_11 = False
    var_12 = []
    var_13 = []
    var_14 = []
    var_15 = True
    var_16 = []
    var_17 = {var_2: var_11, var_3: var_11, var_4: var_12, var_5: var_13, var_6: var_14, var_7: var_15, var_7: var_11, var_8: var_11, var_9: var_14, var_10: var_16}
    var_18 = [var_0, var_1, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'some_module'
    module_0.module_key(var_22, var_21, var_15)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort_straight'
    var_9 = 'length_sort_sections'
    var_10 = module_0.naturally(var_7)
    assert module_0.TYPE_CHECKING is False
    var_11 = 'force_to_top'
    var_12 = []
    var_13 = True
    var_14 = []
    var_15 = {var_2: var_13, var_3: var_13, var_4: var_10, var_5: var_12, var_6: var_10, var_7: var_13, var_0: var_13, var_8: var_13, var_9: var_14, var_11: var_11}
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_1.type(*var_16, **var_17)
    var_19 = var_18()
    module_0.module_key(var_0, var_19, var_19, section_name=var_19)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'Confg'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort'
    var_9 = 'length_sort_straight'
    var_10 = 'force_to_top'
    var_11 = []
    var_12 = []
    var_13 = []
    var_14 = module_0.naturally(var_2)
    assert module_0.TYPE_CHECKING is False
    var_15 = True
    var_16 = []
    var_17 = {var_2: var_15, var_3: var_15, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_15, var_8: var_15, var_9: var_15, var_8: var_11, var_10: var_16}
    var_18 = [var_0, var_1, var_17]
    var_19 = {}
    var_20 = module_1.type(*var_18, **var_19)
    var_21 = var_20()
    var_22 = 'som?e_module'
    module_0.module_key(var_22, var_21, var_15)

def test_case_14():
    var_0 = ()
    var_1 = 'reverse_relative'
    var_2 = 'order_by_type'
    var_3 = 'constants'
    var_4 = 'classes'
    var_5 = 'variables'
    var_6 = 'case_sensitive'
    var_7 = 'length_sort'
    var_8 = 'length_srtstraight'
    var_9 = 'length_sort_sections'
    var_10 = 'force_to_top'
    var_11 = [var_9, var_5, var_2]
    var_12 = []
    var_13 = True
    var_14 = []
    var_15 = {var_1: var_13, var_2: var_13, var_3: var_8, var_4: var_11, var_5: var_12, var_6: var_13, var_7: var_13, var_8: var_13, var_9: var_8, var_10: var_14}
    var_16 = [var_3, var_0, var_15]
    var_17 = {}
    var_18 = module_1.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = module_0.module_key(var_2, var_19, var_13)
    assert var_20 == 'BB13:order_by_type'
    assert module_0.TYPE_CHECKING is False

def test_case_15():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort'
    var_9 = 'length_sort_straight'
    var_10 = module_0.naturally(var_7)
    assert module_0.TYPE_CHECKING is False
    var_11 = 'force_to_top'
    var_12 = []
    var_13 = []
    var_14 = True
    var_15 = []
    var_16 = {var_2: var_14, var_3: var_14, var_4: var_13, var_5: var_12, var_6: var_13, var_7: var_14, var_8: var_14, var_9: var_14, var_5: var_15, var_11: var_11}
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_1.type(*var_17, **var_18)
    var_20 = var_19()
    var_21 = ''
    var_22 = module_0.module_key(var_21, var_20, var_2, straight_import=var_16)
    assert var_22 == 'AC0:'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'order_by_type'
    var_3 = 'constants'
    var_4 = 'classes'
    var_5 = 'variables'
    var_6 = 'case_sensitive'
    var_7 = 'length_sort_straight'
    var_8 = 'length_sort_sections'
    var_9 = module_0.naturally(var_6)
    assert module_0.TYPE_CHECKING is False
    var_10 = []
    var_11 = []
    var_12 = []
    var_13 = True
    var_14 = []
    var_15 = {var_3: var_13, var_2: var_13, var_3: var_10, var_4: var_11, var_5: var_12, var_6: var_13, var_3: var_13, var_7: var_13, var_8: var_14, var_3: var_3}
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_1.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = ''
    module_0.module_key(var_20, var_19, var_8, straight_import=var_15)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'order_by_type'
    var_3 = 'constants'
    var_4 = 'classes'
    var_5 = 'variables'
    var_6 = 'case_sensitive'
    var_7 = 'length_sort'
    var_8 = 'length_sort_straight'
    var_9 = module_2.purge()
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
    var_10 = 'force_to_top'
    var_11 = module_0.naturally(var_4)
    assert module_0.TYPE_CHECKING is False
    var_12 = []
    var_13 = [var_0, var_10]
    var_14 = True
    var_15 = []
    var_16 = {var_7: var_14, var_2: var_14, var_3: var_12, var_4: var_13, var_5: var_13, var_6: var_14, var_7: var_14, var_8: var_14, var_7: var_15, var_10: var_10}
    var_17 = [var_0, var_1, var_16]
    var_18 = {}
    var_19 = module_1.type(*var_17, **var_18)
    var_20 = var_19()
    var_21 = ''
    var_22 = module_0.module_key(var_21, var_20, var_0, straight_import=var_16)
    assert var_22 == 'AC0:'
    var_23 = None
    var_24 = b"\xde \x92'\x0cII\xb2r\x13\xb5K\x01"
    module_0.module_key(var_24, var_20, var_23)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'length_sort'
    var_8 = 'length_sort_sections'
    var_9 = module_0.naturally(var_4)
    assert module_0.TYPE_CHECKING is False
    var_10 = 'force_to_top'
    var_11 = []
    var_12 = True
    var_13 = module_2.purge()
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
    var_14 = {var_2: var_12, var_3: var_12, var_4: var_11, var_5: var_9, var_6: var_9, var_8: var_12, var_7: var_12, var_8: var_12, var_8: var_13, var_10: var_10}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_1.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = '.ZM'
    module_0.module_key(var_19, var_18, var_8, straight_import=var_14)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort'
    var_9 = 'legth_sort_sectioXns'
    var_10 = module_0.naturally(var_7)
    assert module_0.TYPE_CHECKING is False
    var_11 = 'force_to_top'
    var_12 = []
    var_13 = True
    var_14 = module_2.purge()
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
    var_15 = {var_2: var_13, var_3: var_13, var_4: var_12, var_5: var_10, var_6: var_10, var_7: var_13, var_8: var_13, var_9: var_13, var_9: var_14, var_11: var_0, var_6: var_3, var_11: var_11}
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_1.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = ''
    var_21 = module_0.module_key(var_20, var_19, var_9, straight_import=var_15)
    assert var_21 == 'AC0:'
    var_22 = None
    module_0.module_key(var_19, var_19, var_22)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 'Config'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort'
    var_9 = 'length_sort_sections'
    var_10 = module_0.naturally(var_7)
    assert module_0.TYPE_CHECKING is False
    var_11 = 'force_to_top'
    var_12 = []
    var_13 = module_2.purge()
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
    var_14 = {var_2: var_13, var_3: var_13, var_4: var_12, var_5: var_10, var_6: var_10, var_7: var_13, var_8: var_13, var_9: var_13, var_9: var_13, var_11: var_11}
    var_15 = [var_0, var_1, var_14]
    var_16 = {}
    var_17 = module_1.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = '.ZM'
    module_0.module_key(var_19, var_18, var_9, straight_import=var_14)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = '|Zr"bvwI}q6e3\';B+<:'
    var_1 = ()
    var_2 = 'reverse_relative'
    var_3 = 'order_by_type'
    var_4 = 'constants'
    var_5 = 'classes'
    var_6 = 'variables'
    var_7 = 'case_sensitive'
    var_8 = 'length_sort'
    var_9 = 'length_sort_sections'
    var_10 = module_0.naturally(var_7)
    assert module_0.TYPE_CHECKING is False
    var_11 = 'force_to_top'
    var_12 = []
    var_13 = True
    var_14 = module_2.purge()
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
    var_15 = {var_2: var_13, var_3: var_13, var_4: var_12, var_5: var_10, var_6: var_10, var_7: var_13, var_8: var_13, var_9: var_13, var_9: var_14, var_11: var_11}
    var_16 = [var_0, var_1, var_15]
    var_17 = {}
    var_18 = module_1.type(*var_16, **var_17)
    var_19 = var_18()
    var_20 = 'M'
    var_21 = module_0.module_key(var_20, var_19, var_9, straight_import=var_15)
    assert var_21 == 'BB1:M'
    var_22 = module_0.module_key(var_11, var_19, var_19, section_name=var_10)
    assert var_22 == 'AC12:force_to_top'
    var_23 = '\t'
    var_24 = False
    var_25 = False
    var_26 = module_0.module_key(var_23, var_19, var_19, var_24, straight_import=var_25)
    assert var_26 == 'BC1:\t'
    var_27 = var_14.__bool__()
    assert var_27 is False
    var_28 = '"\n:\x0cxFy<vxUggd\x0c'
    var_29 = False
    module_0.module_key(var_28, var_8, var_29)