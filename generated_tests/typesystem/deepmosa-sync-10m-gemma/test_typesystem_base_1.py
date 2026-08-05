# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = module_0.Message(text=var_4, key=var_4, index=var_4, position=var_1, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True

def test_case_1():
    var_0 = 'error 1'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error 1'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 == var_1)
    assert var_2 is True

def test_case_2():
    var_0 = 'xv&d*-'
    var_1 = module_0.BaseError(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1

def test_case_3():
    var_0 = 'eMro{'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'eMro{'
    assert var_1.code == 'eMro{'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 != var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = []
    module_0.ValidationError(messages=var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_7():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None

def test_case_8():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = None
    var_3 = module_0.Message(text=var_2, key=var_2, index=var_2, position=var_1, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.Position'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text=None, code='custom', position=Position(line_no=None, column_no=None, char_index=None))"

def test_case_9():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == -8644591753022417039
    var_3 = bool(var_1 != var_1)

def test_case_10():
    var_0 = 'Field error'
    var_1 = 'field'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Field error'
    assert var_3.code == 'custom'
    assert var_3.index == ['field']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Nested error'
    var_5 = 'parent'
    var_6 = 'child'
    var_7 = [var_5, var_6]
    var_8 = module_0.Message(text=var_4, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Nested error'
    assert var_8.code == 'custom'
    assert var_8.index == ['parent', 'child']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_3, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 2
    var_11 = {var_6: var_4}
    var_12 = {var_1: var_0, var_5: var_11}
    var_13 = str(var_12)
    var_14 = str(var_10)
    var_15 = bool(var_14 == var_13)
    assert var_15 is True

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = var_1.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_4 = None
    var_5 = 'UI,#o'
    var_6 = module_0.ValidationError(text=var_5, position=var_5, messages=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    module_0.ValidationError(text=var_4)

def test_case_12():
    var_0 = '\x0bqh8V'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '\x0bqh8V'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'error 2'
    var_3 = module_0.Message(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error 2'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_1 == var_0)

def test_case_13():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text='error', code='custom')"
    var_4 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = bool(var_1 == var_4)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = False
    var_1 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is False
    assert var_1.code == 'custom'
    assert var_1.index == [False]
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=False, code='custom', index=[False])"
    module_0.ValidationError()

def test_case_15():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success'
    assert var_1.error is None
    var_2 = var_1.__bool__()
    assert var_2 is True
    var_3 = list(var_1)
    var_4 = bool(var_3 == ['success', None])
    assert var_4 is True

def test_case_16():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error'
    assert var_4.code == 'code'
    assert var_4.index == ['key']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = bool(var_6 != 'not an error')
    assert var_7 is True

def test_case_17():
    var_0 = '\tKRSYRh`~|O\t'
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '\tKRSYRh`~|O\t'
    assert var_1.code == 'custom'
    assert var_1.index == '\tKRSYRh`~|O\t'
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_18():
    var_0 = 'error'
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error'
    assert var_3.code == 'custom'
    assert var_3.index == ['a']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'b'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'error'
    assert var_6.code == 'custom'
    assert var_6.index == ['b']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = bool(var_3 != var_6)
    assert var_7 is True

def test_case_19():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success'
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = bool(var_2 == ['success', None])
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = var_1.__bool__()
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_5) == 1
    var_6 = var_5.__str__()
    assert var_6 == "ValuesView(BaseError(text='ValidationResult(value=None)', code='custom'))"
    var_7 = var_4.__eq__(var_5)
    assert var_7 is False
    var_8 = var_4.__str__()
    assert var_8 == 'ValidationResult(value=None)'
    module_0.ValidationError(code=var_0, position=var_5, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = None
    var_2 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = None
    var_7 = var_2.__repr__()
    assert var_7 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_8 = module_0.ValidationResult(value=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == 'Position(line_no=None, column_no=None, char_index=None)'
    assert var_8.error is None
    var_9 = 'Ryz'
    var_10 = module_0.Message(text=var_7, key=var_3, start_position=var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'Position(line_no=None, column_no=None, char_index=None)'
    assert var_10.code == 'custom'
    assert var_10.index == ['Position(line_no=None, column_no=None, char_index=None)']
    assert f'{type(var_10.start_position).__module__}.{type(var_10.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.end_position is None
    var_11 = var_10.__eq__(var_0)
    assert var_11 is False
    var_12 = var_10.__repr__()
    assert var_12 == "Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom', index=['Position(line_no=None, column_no=None, char_index=None)'], start_position=ValidationResult(value=None), end_position=None)"
    var_13 = module_0.BaseError(text=var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = 'UI,#o'
    module_0.ValidationError(text=var_6, code=var_14, key=var_6)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = None
    var_5 = module_0.Message(text=var_4, key=var_4, index=var_4, position=var_1, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_2.__bool__()
    assert var_6 is True
    var_7 = module_0.BaseError(text=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_7.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8) == 1
    var_9 = var_5.__eq__(var_5)
    assert var_9 is True
    var_10 = var_5.__repr__()
    assert var_10 == "Message(text=None, code='custom', position=Position(line_no=None, column_no=None, char_index=None))"
    var_11 = var_7.__eq__(var_1)
    assert var_11 is False
    module_0.ValidationError(code=var_11, messages=var_5)

def test_case_23():
    var_0 = None
    var_1 = None
    var_2 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = False
    var_6 = module_0.Position(var_1, var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is None
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = var_2.__eq__(var_1)
    assert var_7 is False
    var_8 = 'u^\rd}z'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_8, key=var_4, index=var_4, end_position=var_1)

def test_case_24():
    var_0 = 'error 1'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error 1'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'error 2'
    var_3 = module_0.Message(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error 2'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_1 == var_3)

def test_case_25():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Position(var_1, var_1, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = b'c\x1b'
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_7 = None
    var_8 = module_0.Message(text=var_6, position=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Position(line_no=None, column_no=None, char_index=None)'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__repr__()
    assert var_9 == "Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom')"
    var_10 = var_3.__repr__()
    assert var_10 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_11 = var_3.__repr__()
    assert var_11 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_7, position=var_3, end_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    var_5 = var_4.__str__()
    assert var_5 == "ValuesView(BaseError(text='ValidationResult(value=None)', code='custom'))"
    var_6 = var_3.__eq__(var_4)
    assert var_6 is False
    module_0.ValidationError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = 'qf`_|T7W`[>o/'
    module_0.ValidationError(key=var_4, messages=var_3)

def test_case_28():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = var_1.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_4 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = var_1.__repr__()
    assert var_6 == 'Position(line_no=None, column_no=None, char_index=None)'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0, position=var_1, start_position=var_1)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = None
    var_5 = module_0.Message(text=var_4, key=var_4, index=var_4, position=var_1, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_2.__bool__()
    assert var_6 is True
    var_7 = module_0.BaseError(text=var_3, key=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_7.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8) == 1
    var_9 = var_7.__str__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = var_5.__eq__(var_4)
    assert var_10 is False
    var_11 = var_5.__repr__()
    assert var_11 == "Message(text=None, code='custom', position=Position(line_no=None, column_no=None, char_index=None))"
    var_12 = var_7.__eq__(var_8)
    assert var_12 is False
    module_0.ValidationError(text=var_3, code=var_4, messages=var_12)

def test_case_30():
    var_0 = 'error'
    var_1 = 'code'
    var_2 = 'key'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error'
    assert var_4.code == 'code'
    assert var_4.index == ['key']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = bool(var_6 != 'not an error')
    assert var_7 is True
    var_8 = var_6.__hash__()
    assert var_8 == 8936998406272576240
    var_9 = bool(var_6 != 123)
    assert var_9 is True

def test_case_31():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_2, error=var_2)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is False
    assert var_2.char_index is False
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=False, char_index=False)'
    var_4 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is False
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(error=False)'
    var_6 = None
    var_7 = module_0.Message(text=var_3, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Position(line_no=None, column_no=False, char_index=False)'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_4.__bool__()
    assert var_8 is False
    var_9 = var_7.__eq__(var_7)
    assert var_9 is True
    var_10 = var_2.__repr__()
    assert var_10 == 'Position(line_no=None, column_no=False, char_index=False)'
    var_11 = var_7.__repr__()
    assert var_11 == "Message(text='Position(line_no=None, column_no=False, char_index=False)', code='custom')"
    module_0.ValidationError(position=var_6)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = None
    var_7 = module_0.Message(text=var_6, key=var_6, index=var_6, position=var_1, start_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = var_3.__bool__()
    assert var_8 is True
    var_9 = module_0.BaseError(text=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.messages()
    var_11 = var_10.__str__()
    assert var_11 == "[Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom')]"
    var_12 = var_10.__repr__()
    assert var_12 == "[Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom')]"
    module_0.ValidationError(text=var_6, key=var_6, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = None
    var_6 = module_0.Message(text=var_4, index=var_5, end_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__bool__()
    assert var_7 is True
    var_8 = -3535
    var_9 = module_0.BaseError(text=var_2, key=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.values()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_10) == 1
    var_11 = var_10.__str__()
    assert var_11 == "ValuesView(BaseError([Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom', index=[-3535])]))"
    var_12 = var_6.__eq__(var_5)
    assert var_12 is False
    var_13 = var_6.__repr__()
    assert var_13 == "Message(text='ValidationResult(value=None)', code='custom')"
    var_14 = var_10.__contains__(var_10)
    assert var_14 is False
    var_14.__getitem__(var_10)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = None
    var_5 = var_1.__repr__()
    assert var_5 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_6 = module_0.Message(text=var_4, key=var_4, index=var_4, position=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = var_2.__bool__()
    assert var_7 is True
    var_8 = module_0.BaseError(text=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.messages(add_prefix=var_3)
    var_10 = module_0.BaseError(code=var_0, messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_10.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    var_12 = var_8.__str__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = var_6.__eq__(var_0)
    assert var_13 is False
    var_14 = None
    var_15 = var_8.__eq__(var_14)
    assert var_15 is False
    var_16 = var_15.__repr__()
    assert var_16 == 'False'
    var_17 = var_10.__str__()
    assert var_17 == "{'ValidationResult(value=None)': 'ValidationResult(value=None)'}"
    var_18 = var_10.__iter__()
    var_19 = var_18.__repr__()
    var_20 = var_15.__repr__()
    assert var_20 == 'False'
    module_0.ValidationError(key=var_11, position=var_0, messages=var_4)

def test_case_36():
    var_0 = 'Error'
    var_1 = '$r k_0u'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == '$r k_0u'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'err_02'
    var_4 = module_0.Message(text=var_0, code=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'err_02'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

def test_case_37():
    var_0 = 'Error 1'
    var_1 = 'err1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'err1'
    assert var_4.index == ['key1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'err2'
    var_7 = 'key2'
    var_8 = [var_7, var_5]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'err2'
    assert var_9.index == ['key2', 'Error 2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.ValidationError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 2
    var_12 = [var_4, var_9]
    var_13 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 2
    var_14 = bool(var_11 == var_13)
    assert var_14 is True

def test_case_38():
    var_0 = 'Field error'
    var_1 = 'field'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Field error'
    assert var_3.code == 'custom'
    assert var_3.index == ['field']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Nested error'
    var_5 = 'parent'
    var_6 = 'child'
    var_7 = [var_5, var_6]
    var_8 = module_0.Message(text=var_4, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Nested error'
    assert var_8.code == 'custom'
    assert var_8.index == ['parent', 'child']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_3, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 2
    var_11 = {var_6: var_4}
    var_12 = var_10.__repr__()
    assert var_12 == "ValidationError([Message(text='Field error', code='custom', index=['field']), Message(text='Nested error', code='custom', index=['parent', 'child'])])"
    var_13 = {var_1: var_0, var_5: var_11}
    var_14 = str(var_13)
    var_15 = str(var_10)
    var_16 = bool(var_15 == var_14)
    assert var_16 is True

def test_case_39():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 2
    assert var_4.column_no == 2
    assert var_4.char_index == 3
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

def test_case_40():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = module_0.Position(var_0, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 3
    assert var_4.char_index == 3
    var_5 = bool(var_3 != var_4)
    assert var_5 is True

def test_case_41():
    var_0 = 'Error 1'
    var_1 = 'err1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'err1'
    assert var_4.index == ['key1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'err2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'err2'
    assert var_9.index == ['key2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.ValidationError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 2
    var_12 = [var_4, var_9]
    var_13 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 2
    var_14 = bool(var_11 == var_13)
    assert var_14 is True

def test_case_42():
    var_0 = '\tKRS#YRh`~|O\t'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1, key=var_1, index=var_1, position=var_1, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == '\tKRS#YRh`~|O\t'
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_1, code=var_1, index=var_1, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False