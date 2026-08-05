# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0
import builtins as module_1

def test_case_0():
    var_0 = '~w:4-vK'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_1():
    var_0 = ']nclxO%z'
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ']nclxO%z'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position == ']nclxO%z'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = []
    module_0.ParseError(messages=var_0)

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_5():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_6():
    var_0 = 1944
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = None
    var_3 = True
    var_4 = module_0.Position(var_0, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1944
    assert var_4.column_no is None
    assert var_4.char_index is True
    var_5 = var_4.__repr__()
    assert var_5 == 'Position(line_no=1944, column_no=None, char_index=True)'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == -2820639458673338211
    module_0.ValidationError(messages=var_0)

def test_case_8():
    var_0 = 'h 2\nY `3o0JNw6b'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, messages=var_0)

def test_case_9():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_0.__bool__()
    assert var_2 is True

def test_case_10():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is False
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is False
    assert var_3.error is None
    var_4 = var_1.__len__()
    assert var_4 == 1
    var_5 = var_3.__bool__()
    assert var_5 is True
    var_6 = module_0.Message(text=var_2, start_position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is False
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is False
    assert var_6.end_position is None
    var_7 = var_1.__iter__()
    var_8 = var_6.__eq__(var_4)
    assert var_8 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_2, messages=var_4)

def test_case_11():
    var_0 = 'text1'
    var_1 = 'code1'
    var_2 = [var_1, var_0]
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'text1'
    assert var_3.code == 'code1'
    assert var_3.index == ['code1', 'text1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'text2'
    var_5 = 'code2'
    var_6 = 'key'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'text2'
    assert var_8.code == 'code2'
    assert var_8.index == ['key']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_3, var_8]
    var_10 = module_0.BaseError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 2
    var_11 = repr(var_10)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_2 = module_0.BaseError(text=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_2.__contains__(var_0)
    assert var_5 is False
    var_6 = var_3.__eq__(var_3)
    assert var_6 is True
    var_5.__iter__()

def test_case_13():
    var_0 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "BaseError([Message(text='|uy:tnsUU!dz7\\n:t\\t\\rN', code='custom', index=['|uy:tnsUU!dz7\\n:t\\t\\rN'])])"
    var_3 = None
    var_4 = module_0.Message(text=var_3, index=var_3, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False

def test_case_14():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success'
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2

def test_case_15():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is False
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is False
    assert var_3.error is None
    var_4 = var_1.__len__()
    assert var_4 == 1
    var_5 = var_3.__bool__()
    assert var_5 is True
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_2, messages=var_2)

def test_case_16():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_0.__iter__()
    var_3 = var_1.__hash__()
    var_4 = var_1.messages()
    var_5 = var_1.__len__()
    assert var_5 == 1
    var_6 = var_0.__bool__()
    assert var_6 is True
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = var_1.__eq__(var_6)
    assert var_8 is False
    var_9 = None
    var_10 = module_0.ValidationResult(value=var_9, error=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is False
    var_11 = var_1.__len__()
    assert var_11 == 1
    var_12 = var_0.__bool__()
    assert var_12 is True
    var_13 = module_0.Message(text=var_5, code=var_11, key=var_9, position=var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 1
    assert var_13.code == 1
    assert var_13.index == []
    assert var_13.start_position is False
    assert var_13.end_position is False
    var_14 = var_11.__eq__(var_9)
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_17():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_4 = var_1.__str__()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_0.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = var_1.values()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_6) == 1
    var_7 = var_3.__eq__(var_2)
    var_8 = var_6.__str__()
    assert var_8 == 'ValuesView(BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None)))'
    var_9 = var_0.__bool__()
    assert var_9 is True

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    var_3 = var_1.messages()
    var_4 = var_1.__len__()
    assert var_4 == 1
    var_5 = var_0.__bool__()
    assert var_5 is True
    var_6 = var_4.__bool__()
    assert var_6 is True
    var_7 = module_0.ValidationResult(error=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error == 1
    var_4.__len__()

def test_case_19():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.error is None
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_2, messages=var_2)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = None
    var_2 = '\x0bjL.gm+EW_&x\r=-A'
    var_3 = module_0.Message(text=var_2, code=var_0, position=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '\x0bjL.gm+EW_&x\r=-A'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3]
    var_5 = module_0.ParseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_5.__eq__(var_1)
    assert var_6 is False
    var_6.__contains__(var_1)

def test_case_21():
    var_0 = '|uy:tnsUU!d7\n:t\t\rN'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

def test_case_22():
    var_0 = None
    var_1 = 1663
    var_2 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = None
    var_5 = module_0.BaseError(text=var_3, code=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__contains__(var_4)
    assert var_6 is False
    var_7 = var_5.__iter__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_1, position=var_6, start_position=var_1, end_position=var_1)

def test_case_23():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is False
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is False
    assert var_3.error is None
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_2, messages=var_2)

def test_case_24():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_3.__repr__()
    assert var_4 == 'False'
    var_5 = -929
    var_6 = module_0.Position(var_2, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is None
    assert var_6.column_no is None
    assert var_6.char_index == -929
    var_7 = var_1.__eq__(var_2)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = var_1.__len__()
    assert var_9 == 1
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_6, messages=var_3)

def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

def test_case_26():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_2) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_2, position=var_2, end_position=var_2)

def test_case_27():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    var_3 = var_1.messages()
    var_4 = var_1.__len__()
    assert var_4 == 1
    var_5 = var_0.__bool__()
    assert var_5 is True
    var_6 = var_4.__bool__()
    assert var_6 is True
    var_7 = var_1.__eq__(var_5)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_1.__len__()
    assert var_9 == 1
    var_10 = var_9.__bool__()
    assert var_10 is True
    var_11 = var_9.__bool__()
    assert var_11 is True
    with pytest.raises(AssertionError):
        module_0.Message(text=var_7, key=var_11, index=var_4)

def test_case_28():
    var_0 = 'text1'
    var_1 = 'code1'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'text1'
    assert var_3.code == 'code1'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'text2'
    var_5 = 'code2'
    var_6 = 'key'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'text2'
    assert var_8.code == 'code2'
    assert var_8.index == ['key']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_3, var_8]
    var_10 = module_0.BaseError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 2
    var_11 = repr(var_10)

def test_case_29():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is False
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_1, error=var_2)

def test_case_30():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = var_3.__bool__()
    assert var_5 is True
    var_6 = var_2.__repr__()
    assert var_6 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_7 = module_0.BaseError(messages=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_0.__bool__()
    assert var_8 is True
    var_9 = var_0.__bool__()
    assert var_9 is True
    var_10 = module_1.BaseException()
    var_11 = None
    var_12 = var_1.messages(add_prefix=var_3)
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_11)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = var_3.__bool__()
    assert var_5 is True
    var_6 = var_1.__repr__()
    assert var_6 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_7 = var_2.__repr__()
    assert var_7 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_8 = module_0.BaseError(messages=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_0.__bool__()
    assert var_9 is True
    var_10 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_11 = module_0.BaseError(text=var_10, key=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_0.__bool__()
    assert var_12 is True
    var_13 = []
    var_14 = module_1.BaseException(*var_13)
    var_15 = ';\r;O\x0bV$'
    var_16 = None
    var_17 = module_0.Message(text=var_15, key=var_15, index=var_16, start_position=var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == ';\r;O\x0bV$'
    assert var_17.code == 'custom'
    assert var_17.index == [';\r;O\x0bV$']
    assert f'{type(var_17.start_position).__module__}.{type(var_17.start_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17.start_position) == 1
    assert var_17.end_position is None
    var_18 = var_8.values()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_18) == 1
    var_19 = var_11.__str__()
    assert var_19 == "{'|uy:tnsUU!dz7\\n:t\\t\\rN': '|uy:tnsUU!dz7\\n:t\\t\\rN'}"
    var_20 = var_11.items()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_20) == 1
    var_20.items()

def test_case_32():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = var_3.__bool__()
    assert var_5 is True
    var_6 = var_2.__repr__()
    assert var_6 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_7 = module_0.BaseError(messages=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_0.__bool__()
    assert var_8 is True
    var_9 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_10 = module_0.BaseError(text=var_9, key=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_1.__len__()
    assert var_11 == 1
    var_12 = var_11.__bool__()
    assert var_12 is True
    var_13 = module_1.BaseException()
    var_14 = module_0.Message(text=var_3, start_position=var_11, end_position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_14.text).__module__}.{type(var_14.text).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert var_14.start_position == 1
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'builtins.BaseException'
    var_15 = module_0.Message(text=var_9, key=var_6)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == '|uy:tnsUU!dz7\n:t\t\rN'
    assert var_15.code == 'custom'
    assert var_15.index == ['[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]']
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_14.__eq__(var_11)
    assert var_16 is False
    var_17 = var_10.__eq__(var_11)
    assert var_17 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_3, messages=var_10)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = None
    var_6 = module_0.Position(var_4, var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert f'{type(var_6.column_no).__module__}.{type(var_6.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6.column_no) == 1
    assert var_6.char_index is None
    var_7 = var_3.__bool__()
    assert var_7 is True
    var_8 = var_1.__repr__()
    assert var_8 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_9 = var_2.__repr__()
    assert var_9 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_10 = module_0.BaseError(messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_0.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_6.__eq__(var_6)
    assert var_12 is False
    var_13 = var_10.__len__()
    assert var_13 == 1
    var_14 = module_1.BaseException(*var_10)
    var_15 = var_6.__eq__(var_9)
    assert var_15 is False
    var_16 = None
    var_17 = module_0.Message(text=var_16, key=var_13)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text is None
    assert var_17.code == 'custom'
    assert var_17.index == [1]
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = var_17.__eq__(var_16)
    assert var_18 is False
    var_6.__repr__()

def test_case_34():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = None
    var_3 = module_0.Position(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = var_0.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True
    var_6 = var_0.__bool__()
    assert var_6 is True
    var_7 = None
    var_8 = module_0.Message(text=var_7, index=var_7, position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_7)
    assert var_9 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_2, position=var_2, messages=var_7)

def test_case_35():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_0.__bool__()
    assert var_3 is True
    var_4 = None
    var_5 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert f'{type(var_5.column_no).__module__}.{type(var_5.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.column_no) == 1
    assert var_5.char_index is None
    var_6 = var_2.__bool__()
    assert var_6 is True
    var_7 = var_1.__repr__()
    assert var_7 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_8 = var_0.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_5.__eq__(var_5)
    assert var_9 is False
    var_10 = var_0.__bool__()
    assert var_10 is True
    var_11 = None
    var_12 = module_0.Message(text=var_11, index=var_11, position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text is None
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = var_12.__eq__(var_11)
    assert var_13 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_4, position=var_4, messages=var_11)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = None
    var_3 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_4 = module_0.BaseError(text=var_3, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_0.__bool__()
    assert var_5 is True
    var_6 = None
    var_7 = module_0.Message(text=var_6, index=var_6, position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_6)
    assert var_8 is False
    var_9 = var_4.__contains__(var_6)
    assert var_9 is False
    var_10 = module_0.Message(text=var_2, start_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert f'{type(var_10.start_position).__module__}.{type(var_10.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.end_position is None
    var_11 = var_10.__eq__(var_10)
    assert var_11 is True
    var_12 = var_9.__hash__()
    assert var_12 == 0
    var_13 = var_10.__repr__()
    assert var_13 == "Message(text=None, code='custom', start_position=ValidationResult(value=None), end_position=None)"
    var_14 = None
    var_15 = module_0.Message(text=var_14, position=var_6)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text is None
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_4.__len__()
    assert var_16 == 1
    var_16.__contains__(var_16)

def test_case_37():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_0.__bool__()
    assert var_3 is True
    var_4 = None
    var_5 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert f'{type(var_5.column_no).__module__}.{type(var_5.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.column_no) == 1
    assert var_5.char_index is None
    var_6 = var_1.__repr__()
    assert var_6 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_7 = var_2.__repr__()
    assert var_7 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_8 = module_0.BaseError(messages=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_1.__iter__()
    var_10 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_11 = module_0.BaseError(text=var_10, key=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_0.__bool__()
    assert var_12 is True
    var_13 = var_8.__contains__(var_8)
    assert var_13 is False
    var_14 = var_8.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = module_0.Message(text=var_9, index=var_9, position=var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_15.text).__module__}.{type(var_15.text).__qualname__}' == 'builtins.dict_keyiterator'
    assert var_15.code == 'custom'
    assert f'{type(var_15.index).__module__}.{type(var_15.index).__qualname__}' == 'builtins.dict_keyiterator'
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'builtins.dict_keyiterator'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'builtins.dict_keyiterator'
    var_16 = var_0.__bool__()
    assert var_16 is True
    var_17 = var_1.__contains__(var_9)
    assert var_17 is False
    var_18 = var_11.__len__()
    assert var_18 == 1
    var_19 = var_13.__eq__(var_18)
    assert var_19 is False
    var_20 = module_0.Message(text=var_13)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text is False
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = var_13.__eq__(var_9)
    var_22 = var_1.__iter__()
    var_23 = var_22.__hash__()
    var_24 = var_15.__repr__()
    var_25 = None
    var_26 = var_17.__eq__(var_25)
    var_27 = var_9.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = None
    var_5 = module_0.Position(var_1, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.line_no) == 1
    assert f'{type(var_5.column_no).__module__}.{type(var_5.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.column_no) == 1
    assert var_5.char_index is None
    var_6 = var_3.__bool__()
    assert var_6 is True
    var_7 = var_1.__repr__()
    assert var_7 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_8 = var_2.__repr__()
    assert var_8 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_9 = module_0.BaseError(messages=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_1.__hash__()
    var_11 = var_1.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    var_12 = var_5.__eq__(var_5)
    assert var_12 is False
    var_13 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_14 = module_0.BaseError(text=var_13, key=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_0.__bool__()
    assert var_15 is True
    var_16 = var_14.__len__()
    assert var_16 == 1
    var_17 = var_5.__eq__(var_8)
    assert var_17 is False
    var_18 = var_16.__repr__()
    assert var_18 == '1'
    var_19 = var_14.__repr__()
    assert var_19 == "BaseError([Message(text='|uy:tnsUU!dz7\\n:t\\t\\rN', code='custom', index=['|uy:tnsUU!dz7\\n:t\\t\\rN'])])"
    var_20 = None
    var_21 = module_0.Message(text=var_20, index=var_20, position=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text is None
    assert var_21.code == 'custom'
    assert var_21.index == []
    assert var_21.start_position is None
    assert var_21.end_position is None
    var_22 = var_21.__eq__(var_20)
    assert var_22 is False
    var_23 = var_14.__contains__(var_20)
    assert var_23 is False
    var_24 = var_21.__eq__(var_21)
    assert var_24 is True
    var_25 = var_5.__eq__(var_4)
    assert var_25 is False
    var_26 = None
    var_27 = var_9.__eq__(var_23)
    assert var_27 is False
    var_28 = var_9.__len__()
    assert var_28 == 1
    var_29 = module_0.BaseError(text=var_28, code=var_26, messages=var_20)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_29) == 1
    var_23.values()

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = var_3.__bool__()
    assert var_5 is True
    var_6 = var_1.__repr__()
    assert var_6 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_7 = var_2.__repr__()
    assert var_7 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_8 = module_0.BaseError(messages=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_10 = module_0.BaseError(text=var_9, key=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_0.__bool__()
    assert var_11 is True
    var_12 = var_10.__len__()
    assert var_12 == 1
    var_13 = var_10.__repr__()
    assert var_13 == "BaseError([Message(text='|uy:tnsUU!dz7\\n:t\\t\\rN', code='custom', index=['|uy:tnsUU!dz7\\n:t\\t\\rN'])])"
    var_14 = None
    var_15 = module_0.Message(text=var_14, index=var_14, position=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text is None
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_15.__eq__(var_14)
    assert var_16 is False
    var_17 = var_10.__contains__(var_14)
    assert var_17 is False
    var_18 = module_0.Message(text=var_1, start_position=var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_18.text).__module__}.{type(var_18.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_18.text) == 1
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert f'{type(var_18.start_position).__module__}.{type(var_18.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.end_position is None
    var_19 = var_18.__eq__(var_18)
    assert var_19 is False
    var_20 = var_1.__eq__(var_18)
    assert var_20 is False
    var_17.__getitem__(var_17)

def test_case_40():
    var_0 = 'invalid input'
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error == 'invalid input'
    var_2 = repr(var_1)
    var_3 = f'ValidationResult(error={repr(var_0)})'

def test_case_41():
    var_0 = 'Err 1'
    var_1 = 'c1'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Err 1'
    assert var_4.code == 'c1'
    assert var_4.index == ['field1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Err 2'
    var_6 = 'c2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Err 2'
    assert var_9.code == 'c2'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = str(var_11)
    assert var_12 == "{'field1': 'Err 1', 'field2': 'Err 2'}"

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = None
    var_6 = module_0.Position(var_4, var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert f'{type(var_6.column_no).__module__}.{type(var_6.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6.column_no) == 1
    assert var_6.char_index is None
    var_7 = var_3.__bool__()
    assert var_7 is True
    var_8 = var_1.__repr__()
    assert var_8 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_9 = var_2.__repr__()
    assert var_9 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_10 = module_0.BaseError(messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_6.__eq__(var_6)
    assert var_11 is False
    var_12 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_13 = module_0.BaseError(text=var_12, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_0.__bool__()
    assert var_14 is True
    var_15 = var_6.__eq__(var_9)
    assert var_15 is False
    var_16 = var_13.__repr__()
    assert var_16 == "BaseError([Message(text='|uy:tnsUU!dz7\\n:t\\t\\rN', code='custom', index=['|uy:tnsUU!dz7\\n:t\\t\\rN'])])"
    var_17 = None
    var_18 = module_0.Message(text=var_16, key=var_5, position=var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text == "BaseError([Message(text='|uy:tnsUU!dz7\\n:t\\t\\rN', code='custom', index=['|uy:tnsUU!dz7\\n:t\\t\\rN'])])"
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert f'{type(var_18.start_position).__module__}.{type(var_18.start_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_18.start_position) == 1
    assert f'{type(var_18.end_position).__module__}.{type(var_18.end_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_18.end_position) == 1
    var_19 = var_18.__eq__(var_18)
    assert var_19 is False
    var_20 = var_13.__contains__(var_17)
    assert var_20 is False
    var_21 = var_6.__eq__(var_20)
    assert var_21 is False
    var_22 = var_1.__contains__(var_12)
    assert var_22 is False
    var_23 = var_1.__eq__(var_5)
    assert var_23 is False
    var_24 = var_1.__len__()
    assert var_24 == 1
    var_25 = var_20.__eq__(var_19)
    assert var_25 is True
    var_26 = var_22.__eq__(var_19)
    assert var_26 is True
    var_27 = var_0.__repr__()
    assert var_27 == 'ValidationResult(value=None)'
    var_28 = var_0.__hash__()
    var_22.__len__()

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = None
    var_6 = module_0.Position(var_4, var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert f'{type(var_6.column_no).__module__}.{type(var_6.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6.column_no) == 1
    assert var_6.char_index is None
    var_7 = var_3.__bool__()
    assert var_7 is True
    var_8 = var_1.__repr__()
    assert var_8 == 'BaseError(text=ValidationResult(value=None), code=ValidationResult(value=None))'
    var_9 = var_1.__len__()
    assert var_9 == 1
    var_10 = var_2.__repr__()
    assert var_10 == '[Message(text=ValidationResult(value=None), code=ValidationResult(value=None))]'
    var_11 = var_1.__contains__(var_5)
    assert var_11 is False
    var_12 = var_11.__repr__()
    assert var_12 == 'False'
    var_13 = module_0.BaseError(messages=var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_6.__eq__(var_6)
    assert var_14 is False
    var_15 = '|uy:tnsUU!dz7\n:t\t\rN'
    var_16 = module_0.BaseError(text=var_15, key=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = var_0.__bool__()
    assert var_17 is True
    var_18 = []
    var_19 = var_6.__eq__(var_10)
    assert var_19 is False
    var_20 = var_16.__repr__()
    assert var_20 == "BaseError([Message(text='|uy:tnsUU!dz7\\n:t\\t\\rN', code='custom', index=['|uy:tnsUU!dz7\\n:t\\t\\rN'])])"
    var_21 = None
    var_22 = module_0.Message(text=var_21, index=var_21, position=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Message'
    assert var_22.text is None
    assert var_22.code == 'custom'
    assert var_22.index == []
    assert var_22.start_position is None
    assert var_22.end_position is None
    var_23 = var_22.__eq__(var_21)
    assert var_23 is False
    var_24 = var_16.__contains__(var_21)
    assert var_24 is False
    var_25 = var_24.__eq__(var_21)
    var_26 = module_0.Message(text=var_24, key=var_17)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.Message'
    assert var_26.text is False
    assert var_26.code == 'custom'
    assert var_26.index == [True]
    assert var_26.start_position is None
    assert var_26.end_position is None
    var_27 = var_11.__eq__(var_21)
    var_28 = var_6.__eq__(var_21)
    assert var_28 is False
    var_29 = var_0.__repr__()
    assert var_29 == 'ValidationResult(value=None)'
    var_30 = None
    var_31 = module_0.ValidationError(text=var_3, code=var_3, messages=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_31) == 1
    var_32 = var_1.__eq__(var_31)
    assert var_32 is True
    var_16.__contains__(var_18)