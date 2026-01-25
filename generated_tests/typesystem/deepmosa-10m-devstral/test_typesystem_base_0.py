# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0
import builtins as module_1

def test_case_0():
    var_0 = "a&V&{K9+e7)#p,'jlF%"
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == "a&V&{K9+e7)#p,'jlF%"
    assert var_1.code == 'custom'
    assert var_1.index == "a&V&{K9+e7)#p,'jlF%"
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_1():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'Error'
    assert var_2.index == ['field']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = bool(not var_2 == 'not a message')
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == "*@bBbfg'W=CEh!`9"
    assert var_1.code == "*@bBbfg'W=CEh!`9"
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Message(text="*@bBbfg\'W=CEh!`9", code="*@bBbfg\'W=CEh!`9")'
    var_3 = None
    module_0.ValidationError(position=var_3, messages=var_3)

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'V9t\\t\\[%C[+~Q'
    module_0.ParseError(key=var_0, messages=var_0)

def test_case_5():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_6():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'Yhkos4Q\r1yQs\t\\p'
    var_1 = None
    var_2 = [var_0, var_0]
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2, start_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Yhkos4Q\r1yQs\t\\p'
    assert var_3.code == 'custom'
    assert var_3.index == ['Yhkos4Q\r1yQs\t\\p', 'Yhkos4Q\r1yQs\t\\p']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__hash__()
    assert var_4 == -2553964486561002048
    module_0.ParseError(messages=var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = 'g6&SZ(4#IO^\x0cmWf`%W'
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert var_3 == 'g6&SZ(4#IO^\x0cmWf`%W'
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_2.__iter__()
    var_6 = var_5.__eq__(var_0)
    var_7 = var_5.__eq__(var_5)
    assert var_7 is True
    var_4.__len__()

def test_case_9():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_6 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = [var_2, var_2, var_2]
    var_8 = module_0.ValidationError(messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = module_0.ValidationError(text=var_0, position=var_6, messages=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_6.__bool__()
    assert var_10 is True

def test_case_10():
    var_0 = '?,PO'
    var_1 = module_0.BaseError(text=var_0, code=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = {}
    module_0.ValidationError(code=var_0, key=var_0, messages=var_1)

def test_case_12():
    var_0 = 'fld'
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'fld'
    assert var_1.code == 'fld'
    assert var_1.index == ['fld']
    assert var_1.start_position is None
    assert var_1.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "*@bBbfg'W=CEh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_4.column_no is True
    assert var_4.char_index is True
    var_5 = var_2.__eq__(var_1)
    assert var_5 is False
    var_6 = module_0.ParseError(text=var_0, code=var_0, messages=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_2.__repr__()
    assert var_7 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    var_8 = var_6.__eq__(var_1)
    assert var_8 is False
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    module_0.ValidationError(key=var_8, position=var_4)

def test_case_14():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = var_0.__repr__()
    assert var_1 == '"*@bBbfg\'W=CEh!`9"'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_0, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__str__()
    assert var_4 == "*@bBbfg'W=CEh!`9"
    var_5 = var_3.__repr__()
    assert var_5 == 'ValidationError(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'

def test_case_15():
    var_0 = False
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is False
    assert var_1.column_no is False
    assert var_1.char_index is False
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_16():
    var_0 = None
    var_1 = '\r8\x0c2x|t%BQgy?:+]lc"'
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False
    var_4 = var_3.__repr__()
    assert var_4 == 'False'

def test_case_17():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "*@bBbfg'W=CEh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_4.column_no is True
    assert var_4.char_index is True
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_7 = var_2.__repr__()
    assert var_7 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    var_8 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is True
    assert var_8.error is None
    var_9 = module_0.ValidationError(text=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_9.__len__()
    assert var_10 == 1

def test_case_18():
    var_0 = 2
    var_1 = 3
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 2
    assert var_2.column_no == 2
    assert var_2.char_index == 3
    var_3 = 5
    var_4 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 2
    assert var_4.column_no == 5
    assert var_4.char_index == 2
    var_5 = bool(var_2 == var_4)
    var_6 = bool(not var_2 == var_4)
    assert var_6 is True
    var_7 = bool(not var_2 == 'not a Position')
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = "*@bBbfg'7W=CEh!`9"
    var_2 = None
    var_3 = module_0.Message(text=var_1, key=var_0, index=var_2, position=var_2, start_position=var_2, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "*@bBbfg'7W=CEh!`9"
    assert var_3.code == 'custom'
    assert f'{type(var_3.index).__module__}.{type(var_3.index).__qualname__}' == 'builtins.list'
    assert len(var_3.index) == 1
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == 'Message(text="*@bBbfg\'7W=CEh!`9", code=\'custom\', index=[ValidationResult(value=None)])'
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    module_0.ValidationError(position=var_2)

def test_case_20():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "*@bBbfg'W=CEh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_4.column_no is True
    assert var_4.char_index is True
    var_5 = var_4.__repr__()
    assert var_5 == 'Position(line_no=Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\'), column_no=True, char_index=True)'
    var_6 = module_0.ValidationError(text=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_4.__eq__(var_2)
    assert var_7 is False

def test_case_21():
    var_0 = True
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is True
    assert var_1.error is None

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = 'Vzttt\\[%C[+~Q'
    var_2 = module_0.Message(text=var_1, code=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Vzttt\\[%C[+~Q'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    var_4 = var_2.__eq__(var_0)
    assert var_4 is False
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text='Vzttt\\\\[%C[+~Q', code='custom')"
    var_6 = '?,PO'
    var_7 = '$a'
    var_8 = module_0.BaseError(text=var_6, code=var_7, position=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__hash__()
    assert var_9 == -1932823590818157061
    var_10 = var_9.__repr__()
    assert var_10 == '-1932823590818157061'
    var_11 = var_2.__hash__()
    assert var_11 == -8087026713830849668
    module_0.ValidationError(code=var_1, position=var_0)

def test_case_23():
    var_0 = "*@bBbfg'W=CEh!`9"
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, position=var_0, start_position=var_0, end_position=var_0)

def test_case_24():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = "E^+%O3V'Ru(alA"
    var_2 = None
    var_3 = module_0.Message(text=var_1, key=var_2, position=var_2, start_position=var_2, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "E^+%O3V'Ru(alA"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True

def test_case_25():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_6 = [var_2, var_2, var_2]
    var_7 = module_0.ValidationError(messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1

def test_case_26():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = var_0.__hash__()
    assert var_2 == -5973083950762143840
    var_3 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "*@bBbfg'W=CEh!`9"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    var_5 = True
    var_6 = module_0.Position(var_3, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is True
    assert var_6.char_index is True
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_3.__eq__(var_1)
    assert var_8 is False
    var_9 = var_3.__repr__()
    assert var_9 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    var_10 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = module_0.ValidationError(text=var_9, key=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_6.__eq__(var_1)
    assert var_12 is False

def test_case_27():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = "*@bBbfg'W=CEh!`9"
    var_2 = None
    var_3 = module_0.Message(text=var_1, key=var_2, position=var_2, start_position=var_2, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "*@bBbfg'W=CEh!`9"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = True
    var_5 = module_0.Position(var_3, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_5.column_no is True
    assert var_5.char_index is True
    var_6 = False
    var_7 = module_0.Position(var_2, var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is False
    assert var_7.char_index is None
    var_8 = module_0.BaseError(text=var_1, code=var_5, key=var_4, position=var_2, messages=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__iter__()
    var_10 = module_0.Message(text=var_2, end_position=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert f'{type(var_10.end_position).__module__}.{type(var_10.end_position).__qualname__}' == 'builtins.dict_keyiterator'
    var_11 = var_10.__eq__(var_2)
    assert var_11 is False
    var_12 = var_3.__repr__()
    assert var_12 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_8, error=var_9)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = True
    var_3 = module_0.Message(text=var_0, key=var_1, position=var_2, start_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "*@bBbfg'W=CEh!`9"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is True
    assert var_3.end_position is True
    var_4 = True
    var_5 = True
    var_6 = module_0.Position(var_5, var_4, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no is True
    assert var_6.char_index is None
    var_7 = var_3.__eq__(var_1)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_3.__eq__(var_1)
    assert var_9 is False
    var_10 = var_3.__repr__()
    assert var_10 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\', position=True)'
    var_11 = module_0.ValidationResult(value=var_4, error=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is True
    assert var_11.error is None
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "*@bBbfg'W=CEh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationError(text=var_0, code=var_1, key=var_1, messages=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_1)
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_7 = var_3.__eq__(var_4)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = var_4.__eq__(var_7)
    var_10 = var_4.__repr__()
    assert var_10 == 'KeysView(ValidationError(text="*@bBbfg\'W=CEh!`9", code=\'custom\'))'
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    module_0.ValidationError(text=var_7, position=var_4, messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = None
    var_2 = 'C-O\x0c^\r^8Jw3\x0cJ"'
    var_3 = module_0.Message(text=var_2, start_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'C-O\x0c^\r^8Jw3\x0cJ"'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__hash__()
    assert var_4 == -8087026713830849668
    var_5 = '%f6`PuDkl^q# y%R0*-'
    var_6 = module_0.Message(text=var_5, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == '%f6`PuDkl^q# y%R0*-'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.end_position is None
    var_7 = var_6.__repr__()
    assert var_7 == "Message(text='%f6`PuDkl^q# y%R0*-', code='custom', start_position=ValidationResult(value=None), end_position=None)"
    var_8 = True
    var_9 = module_0.Position(var_1, var_8, var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no is True
    assert var_9.char_index is None
    var_10 = var_9.__eq__(var_1)
    assert var_10 is False
    var_11 = var_3.__eq__(var_1)
    assert var_11 is False
    var_12 = var_6.__repr__()
    assert var_12 == "Message(text='%f6`PuDkl^q# y%R0*-', code='custom', start_position=ValidationResult(value=None), end_position=None)"
    var_13 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    module_0.ValidationError(text=var_1, code=var_7, key=var_10, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_4.column_no is True
    assert var_4.char_index is True
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_7 = var_4.__repr__()
    assert var_7 == 'Position(line_no=Message(text="bBg\'W=Eh!`9", code=\'custom\'), column_no=True, char_index=True)'
    var_8 = var_2.__eq__(var_1)
    assert var_8 is False
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_10 = [var_2, var_2, var_2]
    var_11 = var_4.__repr__()
    assert var_11 == 'Position(line_no=Message(text="bBg\'W=Eh!`9", code=\'custom\'), column_no=True, char_index=True)'
    module_0.ValidationError(text=var_1, code=var_7, position=var_1, messages=var_10)

def test_case_32():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "*@bBbfg'W=CEh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_4.column_no is True
    assert var_4.char_index is True
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_7 = var_4.__repr__()
    assert var_7 == 'Position(line_no=Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\'), column_no=True, char_index=True)'
    var_8 = var_2.__repr__()
    assert var_8 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    var_9 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = [var_2, var_2, var_2]
    var_11 = module_0.ValidationError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_2.__hash__()
    assert var_12 == -8087026713830849668
    var_13 = var_11.__eq__(var_1)
    assert var_13 is False
    var_14 = var_11.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14) == 1
    var_15 = var_14.__repr__()
    assert var_15 == 'ItemsView(ValidationError([Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\'), Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\'), Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')]))'

def test_case_33():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = False
    var_4 = module_0.Position(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_5.column_no is False
    assert var_5.char_index is False
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = var_4.__repr__()
    assert var_7 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_8 = var_2.__eq__(var_1)
    assert var_8 is False
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_10 = var_2.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = module_0.ValidationError(text=var_0, position=var_11, messages=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = module_0.ValidationError(text=var_10, key=var_8, position=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.messages(add_prefix=var_1)

def test_case_34():
    var_0 = "*@bBbfg'W=CEh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "*@bBbfg'W=CEh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_4.column_no is True
    assert var_4.char_index is True
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_7 = var_4.__repr__()
    assert var_7 == 'Position(line_no=Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\'), column_no=True, char_index=True)'
    var_8 = var_2.__repr__()
    assert var_8 == 'Message(text="*@bBbfg\'W=CEh!`9", code=\'custom\')'
    var_9 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = [var_2, var_2, var_2]
    var_11 = module_0.ValidationError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_11.__len__()
    assert var_12 == 1
    var_13 = var_11.__eq__(var_11)
    assert var_13 is True

def test_case_35():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == -8087026713830849668
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_2.__hash__()
    assert var_6 == -8087026713830849668
    var_7 = module_0.Position(var_2, var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_7.column_no is False
    assert var_7.char_index is False
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = var_2.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = var_2.__repr__()
    assert var_11 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_12 = [var_2, var_2, var_2, var_2]
    var_13 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.__str__()
    assert var_14 == '{\'\': "bBg\'W=Eh!`9"}'
    var_15 = module_0.ValidationError(text=var_11, key=var_9, position=var_7)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = var_2.__hash__()
    assert var_16 == -8087026713830849668

def test_case_36():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = False
    var_4 = module_0.Position(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = var_2.__hash__()
    assert var_5 == -8087026713830849668
    var_6 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is False
    assert var_6.char_index is False
    var_7 = var_4.__repr__()
    assert var_7 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_8 = var_2.__eq__(var_1)
    assert var_8 is False
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_10 = var_2.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = module_0.ValidationError(text=var_0, position=var_11, messages=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = module_0.ValidationError(text=var_10, key=var_8, position=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = var_12.__eq__(var_1)
    assert var_15 is False
    var_16 = var_13.messages()
    var_17 = var_14.__repr__()
    assert var_17 == 'KeysView(ValidationError([Message(text=\'Message(text="bBg\\\'W=Eh!`9", code=\\\'custom\\\')\', code=\'custom\', index=[False], position=Position(line_no=Message(text="bBg\'W=Eh!`9", code=\'custom\'), column_no=False, char_index=False))]))'

def test_case_37():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "bBg'W=Eh!`9"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = module_0.Position(var_3, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is False
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_5.__repr__()
    assert var_8 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_9 = var_3.__eq__(var_1)
    assert var_9 is False
    var_10 = var_3.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = var_3.__repr__()
    assert var_11 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_12 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = [var_3, var_3, var_3]
    var_14 = module_0.ValidationError(messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = module_0.ValidationError(text=var_0, position=var_12, messages=var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = var_15.__eq__(var_1)
    assert var_16 is False
    var_17 = var_15.messages(add_prefix=var_2)
    var_18 = var_12.__repr__()
    assert var_18 == 'ValidationResult(value=None)'
    var_19 = var_15.messages(add_prefix=var_18)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "bBg'W=Eh!`9"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = module_0.Position(var_3, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is False
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_5.__repr__()
    assert var_8 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_9 = var_3.__eq__(var_1)
    assert var_9 is False
    var_10 = var_3.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = var_3.__repr__()
    assert var_11 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_12 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    module_0.ValidationError(key=var_1, position=var_6, messages=var_8)

def test_case_39():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = False
    var_4 = module_0.Position(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_5.column_no is False
    assert var_5.char_index is False
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = var_2.__eq__(var_1)
    assert var_7 is False
    var_8 = var_2.__repr__()
    assert var_8 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_10 = [var_2, var_2, var_2]
    var_11 = module_0.ValidationError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_11.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12) == 1
    var_13 = module_0.ValidationError(text=var_9, key=var_7, position=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_11.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = module_0.ValidationResult()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is None
    var_16 = module_0.ValidationResult(value=var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16.value) == 1
    assert var_16.error is None
    var_17 = module_0.ValidationResult(error=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert f'{type(var_17.error).__module__}.{type(var_17.error).__qualname__}' == 'typesystem.base.ValidationResult'
    var_18 = var_17.__repr__()
    assert var_18 == 'ValidationResult(error=ValidationResult(value=ItemsView(ValidationError([Message(text="bBg\'W=Eh!`9", code=\'custom\'), Message(text="bBg\'W=Eh!`9", code=\'custom\'), Message(text="bBg\'W=Eh!`9", code=\'custom\')]))))'

def test_case_40():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = False
    var_4 = module_0.Position(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_5.column_no is False
    assert var_5.char_index is False
    var_6 = var_4.__eq__(var_5)
    assert var_6 is False
    var_7 = var_2.__eq__(var_1)
    assert var_7 is False
    var_8 = var_2.__repr__()
    assert var_8 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_10 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = [var_2, var_2, var_2]
    var_12 = module_0.ValidationError(messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_12.items()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_13) == 1
    var_14 = module_0.ValidationError(text=var_9, key=var_7, position=var_5)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_12.keys()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_15) == 1
    var_16 = var_13.__len__()
    assert var_16 == 1
    var_17 = var_16.__eq__(var_15)
    var_18 = var_12.messages()
    var_19 = var_16.__repr__()
    assert var_19 == '1'

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == -8087026713830849668
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_2.__hash__()
    assert var_6 == -8087026713830849668
    var_7 = module_0.Position(var_2, var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_7.column_no is False
    assert var_7.char_index is False
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_5.__repr__()
    assert var_9 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_10 = '0'
    var_11 = var_2.__eq__(var_1)
    assert var_11 is False
    var_12 = var_2.__repr__()
    assert var_12 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_13 = var_2.__repr__()
    assert var_13 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_14 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = module_0.ValidationError(text=var_13, key=var_10)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = var_15.__str__()
    assert var_16 == '{\'0\': \'Message(text="bBg\\\'W=Eh!`9", code=\\\'custom\\\')\'}'
    var_17 = var_15.__repr__()
    assert var_17 == 'ValidationError([Message(text=\'Message(text="bBg\\\'W=Eh!`9", code=\\\'custom\\\')\', code=\'custom\', index=[\'0\'])])'
    var_18 = '49;5'
    module_0.ValidationError(code=var_18)

def test_case_42():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == -8087026713830849668
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_2.__hash__()
    assert var_6 == -8087026713830849668
    var_7 = module_0.Position(var_2, var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_7.column_no is False
    assert var_7.char_index is False
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_5.__repr__()
    assert var_9 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_2.__repr__()
    assert var_11 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_12 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = [var_2]
    var_14 = module_0.ValidationError(messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_14.__str__()
    assert var_15 == "bBg'W=Eh!`9"
    var_16 = var_14.__repr__()
    assert var_16 == 'ValidationError(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_17 = module_0.ValidationError(text=var_0, position=var_12, messages=var_1)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17) == 1
    var_18 = var_17.__len__()
    assert var_18 == 1
    var_19 = module_0.ValidationError(text=var_11, key=var_10, position=var_7)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19) == 1
    var_20 = var_17.__hash__()
    assert var_20 == 555698486986393479
    var_21 = var_17.__str__()
    assert var_21 == "bBg'W=Eh!`9"
    var_22 = var_17.__iter__()
    var_23 = var_19.__eq__(var_17)
    assert var_23 is False

def test_case_43():
    var_0 = "bB''=Eh9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bB''=Eh9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == -8087026713830849668
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_2.__hash__()
    assert var_6 == -8087026713830849668
    var_7 = module_0.Position(var_2, var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_7.column_no is False
    assert var_7.char_index is False
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_5.__repr__()
    assert var_9 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_11.__iter__()
    var_13 = module_0.ValidationError(text=var_0, position=var_11, messages=var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.__len__()
    assert var_14 == 1
    var_15 = module_0.ValidationError(text=var_0, key=var_10, position=var_7)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = var_13.__hash__()
    assert var_16 == 555698486986393479
    var_17 = var_13.__str__()
    assert var_17 == "bB''=Eh9"
    var_18 = var_13.__iter__()
    var_19 = var_15.__eq__(var_13)
    assert var_19 is False

def test_case_44():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == -8087026713830849668
    var_4 = False
    var_5 = module_0.Position(var_4, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_2.__hash__()
    assert var_6 == -8087026713830849668
    var_7 = var_5.__eq__(var_5)
    assert var_7 is True
    var_8 = var_5.__repr__()
    assert var_8 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = var_2.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = [var_2]
    var_13 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.__str__()
    assert var_14 == "bBg'W=Eh!`9"
    var_15 = var_13.__repr__()
    assert var_15 == 'ValidationError(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_16 = var_13.messages()
    var_17 = module_0.ValidationError(text=var_0, position=var_11, messages=var_1)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17) == 1
    var_18 = var_17.__len__()
    assert var_18 == 1
    var_19 = var_17.__hash__()
    assert var_19 == 555698486986393479
    var_20 = var_17.__str__()
    assert var_20 == "bBg'W=Eh!`9"
    var_21 = var_17.__iter__()
    var_22 = var_13.__eq__(var_17)
    assert var_22 is False

def test_case_45():
    var_0 = 'Error'
    var_1 = 'field1'
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'Error'
    assert var_2.index == ['field1']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'field2'
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'field1'
    assert var_4.index == ['field2']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

def test_case_46():
    var_0 = "bBg'W=Eh!`9"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "bBg'W=Eh!`9"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = False
    var_4 = module_0.Position(var_3, var_1, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = var_2.__hash__()
    assert var_5 == -8087026713830849668
    var_6 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is False
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_4.__repr__()
    assert var_8 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = var_2.__repr__()
    assert var_10 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_11 = var_2.__repr__()
    assert var_11 == 'Message(text="bBg\'W=Eh!`9", code=\'custom\')'
    var_12 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_1, position=var_6, end_position=var_6)

def test_case_47():
    var_0 = 'm!MvbCT'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0)

def test_case_48():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'username'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 1'
    assert var_6.code == 'code1'
    assert var_6.index == ['users', 0, 'username']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = 1
    var_10 = 'email'
    var_11 = [var_2, var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'Error 2'
    assert var_12.code == 'code2'
    assert var_12.index == ['users', 1, 'email']
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = [var_6, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_14._messages
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_14._messages[0].text
    assert var_17 == 'Error 1'
    var_18 = var_14._messages[0].code
    assert var_18 == 'code1'
    var_19 = bool(var_14._messages[0].index == ['users', 0, 'username'])
    assert var_19 is True
    var_20 = var_14._messages[1].text
    assert var_20 == 'Error 2'
    var_21 = var_14._messages[1].code
    assert var_21 == 'code2'
    var_22 = bool(var_14._messages[1].index == ['users', 1, 'email'])
    assert var_22 is True
    var_23 = dict(var_14)

def test_case_49():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__bool__()
    assert var_2 is True
    var_3 = var_1.__iter__()
    var_4 = module_1.Exception(*var_3)
    with pytest.raises(AssertionError):
        module_0.BaseError()