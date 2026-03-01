# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.ValidationError(messages=var_0)

def test_case_2():
    var_0 = 'h 2\nY `3o0JNw6b'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, messages=var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 4632.06445
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == pytest.approx(4632.06445, abs=0.01, rel=0.01)
    assert var_1.error is None
    var_2 = "'uQ#A\x0b$0&C%L!K&W_h|"
    module_0.ParseError(code=var_2)

def test_case_5():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

def test_case_6():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is True
    assert var_2.column_no is True
    assert var_2.char_index == {}
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

def test_case_7():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is True
    assert var_2.column_no is True
    assert var_2.char_index == {}
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=True, column_no=True, char_index={})'
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = -300
    var_1 = None
    var_2 = 'o*/4'
    var_3 = module_0.ValidationError(text=var_2, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    var_5 = var_3.__contains__(var_0)
    assert var_5 is False
    var_5.__len__()

def test_case_9():
    var_0 = None
    var_1 = 'o*/4'
    var_2 = module_0.ValidationError(text=var_1, code=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'l5mZCkW V|N<O_C,'
    module_0.ParseError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.__hash__()
    assert var_4 == -3870499989249068353
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = '4de3BEs&jy;'
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_5.__iter__()
    var_7 = var_2.__hash__()
    assert var_7 == -3870499989249068353
    module_0.ValidationError()

def test_case_13():
    var_0 = None
    var_1 = 'iq*'
    var_2 = module_0.ValidationError(text=var_1, code=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert var_3 == 'iq*'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'ay)IG?'
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Position(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = module_0.Message(text=var_3, index=var_3, position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__str__()
    assert var_6 == 'ay)IG?'
    var_7 = 'R'
    var_8 = module_0.Message(text=var_7, index=var_3, position=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'R'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_4.__repr__()
    assert var_9 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_10 = None
    var_11 = var_5.__eq__(var_3)
    assert var_11 is False
    var_12 = module_0.ValidationResult(value=var_11, error=var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is False
    assert var_12.error is None
    var_13 = var_12.__iter__()
    var_14 = var_2.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14) == 1
    var_15 = var_4.__repr__()
    assert var_15 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_16 = var_2.messages()
    module_0.ValidationError()

def test_case_15():
    var_0 = 'qRwT.W'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_5 = var_4.__str__()
    assert var_5 == '1'
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False

def test_case_16():
    var_0 = 'qRwT.W'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_5 = var_4.__str__()
    assert var_5 == '1'

def test_case_17():
    var_0 = 'minln>>gh'
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'minln>>gh'
    assert var_1.code == 'minln>>gh'
    assert var_1.index == ['minln>>gh']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = module_0.Message(text=var_0, index=var_0, start_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_3.end_position is None
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = None
    var_6 = var_3.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_7.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_7.__iter__()
    var_10 = var_3.__hash__()
    assert var_10 == -3870499989249068353
    var_11 = module_0.Position(var_5, var_5, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert var_11.column_no is None
    assert var_11.char_index is None
    var_12 = '1mdY11\\="HhXcB?`!'
    var_13 = module_0.BaseError(text=var_12, messages=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_13.__iter__()
    var_14.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = True
    var_1 = {var_0: var_0}
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is True
    assert var_2.column_no is True
    assert var_2.char_index == {True: True}
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=True, column_no=True, char_index={True: True})'
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.ValidationError(text=var_3, position=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "ValidationError(text='Position(line_no=True, column_no=True, char_index={True: True})', code='custom')"
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_4 = module_0.Position(var_0, var_0, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert f'{type(var_4.char_index).__module__}.{type(var_4.char_index).__qualname__}' == 'typesystem.base.Message'
    var_5 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = module_0.Message(text=var_0, start_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_6.end_position is None
    var_7 = var_1.__repr__()
    assert var_7 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_8 = var_1.__eq__(var_4)
    assert var_8 is False
    var_9 = var_6.__repr__()
    assert var_9 == "Message(text=None, code='custom', start_position=Position(line_no=None, column_no=None, char_index=None), end_position=None)"
    var_10 = var_5.__hash__()
    assert var_10 == -3870499989249068353
    module_0.ValidationError(code=var_0, key=var_0)

def test_case_21():
    var_0 = None
    var_1 = 'iq*'
    var_2 = module_0.ValidationError(text=var_1, code=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ValidationError(text='iq*', code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = '4de3BEs&jy;'
    var_4 = module_0.ParseError(text=var_2, code=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_4) == 1
    var_4.__getitem__(var_0)

def test_case_23():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is True
    assert var_2.column_no is True
    assert var_2.char_index == {}
    var_3 = module_0.Message(text=var_0, code=var_0, index=var_0, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is True
    assert var_3.code is True
    assert var_3.index is True
    assert var_3.start_position is None
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.Position'
    var_4 = var_3.__repr__()
    assert var_4 == 'Message(text=True, code=True, index=True)'
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__iter__()
    var_7 = var_4.__bool__()
    assert var_7 is True
    module_0.ValidationError(code=var_0)

def test_case_25():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is True
    assert var_2.column_no is True
    assert var_2.char_index == {}
    var_3 = module_0.Message(text=var_0, code=var_0, index=var_0, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is True
    assert var_3.code is True
    assert var_3.index is True
    assert var_3.start_position is None
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.Position'
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = None
    var_2 = 'o*/4'
    var_3 = None
    var_4 = module_0.ValidationError(text=var_2, code=var_3, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__hash__()
    assert var_5 == -5196263335579880431
    var_6 = var_4.__contains__(var_1)
    assert var_6 is False
    var_7 = var_6.__eq__(var_0)
    var_8 = module_0.Position(var_7, var_0, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.line_no).__module__}.{type(var_8.line_no).__qualname__}' == 'builtins.NotImplementedType'
    assert var_8.column_no is None
    assert var_8.char_index is None
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_6.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = True
    var_2 = {}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={})'
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_11.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12) == 1
    var_13 = var_12.__contains__(var_9)
    assert var_13 is False
    var_13.items()

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'ay)IG?'
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = 'g].ku&5ZRp~V-_l'
    var_5 = module_0.Message(text=var_4, key=var_3, position=var_1, start_position=var_3, end_position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'g].ku&5ZRp~V-_l'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = module_0.Message(text=var_1, key=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_2.__contains__(var_3)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = var_5.__eq__(var_6)
    assert var_9 is False
    var_10 = var_7.__repr__()
    assert var_10 == 'False'
    var_7.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = module_0.Message(text=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, messages=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_9.__repr__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = var_9.__bool__()
    assert var_13 is True
    var_14 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = var_11.__repr__()
    assert var_15 == "ValidationError(text='Position(line_no=True, column_no=True, char_index={None: True})', code='custom')"
    var_16 = var_11.__eq__(var_0)
    assert var_16 is False
    var_17 = var_16.__str__()
    assert var_17 == 'False'
    module_0.ValidationError(text=var_0, code=var_16, key=var_0, messages=var_16)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_3 = None
    var_4 = 'R'
    var_5 = module_0.Message(text=var_0, position=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position == 'R'
    assert var_5.end_position == 'R'
    var_6 = var_1.__repr__()
    assert var_6 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_7 = '4de3BEs&jy;'
    var_8 = var_7.__eq__(var_3)
    var_9 = var_5.__repr__()
    assert var_9 == "Message(text=None, code='custom', position='R')"
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=True, code='custom')"
    var_7 = var_4.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Message(text=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_4.__repr__()
    assert var_11 == "Message(text=True, code='custom')"
    var_12 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_8.__repr__()
    assert var_14 == "Message(text=None, code='custom')"
    var_15 = var_10.__iter__()
    var_16 = var_4.__hash__()
    assert var_16 == -3870499989249068353
    var_17 = var_3.__repr__()
    assert var_17 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_18 = var_12.messages(add_prefix=var_6)
    module_0.ValidationError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_3.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult(value=var_11, error=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_13.value).__module__}.{type(var_13.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.value) == 1
    assert var_13.error is None
    var_14 = var_4.__hash__()
    assert var_14 == -3870499989249068353
    module_0.ValidationError(key=var_6, messages=var_9)

def test_case_33():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = -1649
    var_3 = False
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == -1649
    assert var_4.column_no is False
    assert var_4.char_index is False
    var_5 = 'A}da\\RLE"g{70dJvZ'
    var_6 = module_0.Message(text=var_0, code=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'A}da\\RLE"g{70dJvZ'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_1.__repr__()
    assert var_7 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_8 = var_6.__eq__(var_0)
    assert var_8 is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, position=var_4, start_position=var_4)

def test_case_34():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = module_0.Message(text=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = var_3.__eq__(var_0)
    assert var_11 is False
    var_12 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = var_4.__hash__()
    assert var_13 == -3870499989249068353
    var_14 = []
    var_15 = module_0.ValidationError(text=var_14, key=var_5)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = var_15.__repr__()
    assert var_16 == "ValidationError([Message(text=[], code='custom', index=['Position(line_no=True, column_no=True, char_index={None: True})'])])"

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_3.__eq__(var_0)
    assert var_12 is False
    var_13 = var_11.__len__()
    assert var_13 == 1
    var_14 = module_0.ValidationResult(value=var_11, error=var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_14.value).__module__}.{type(var_14.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14.value) == 1
    assert var_14.error is None
    var_15 = var_11.__repr__()
    assert var_15 == "ValidationError(text='Position(line_no=True, column_no=True, char_index={None: True})', code='custom')"
    var_16 = var_7.__hash__()
    assert var_16 == -3870499989249068353
    var_17 = var_11.__len__()
    assert var_17 == 1
    module_0.ValidationError(code=var_0, position=var_17, messages=var_17)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = ''
    var_3 = module_0.Message(text=var_2, code=var_2, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == ''
    assert var_3.index == []
    assert var_3.start_position == ''
    assert var_3.end_position == ''
    module_0.ValidationError(messages=var_2)

def test_case_37():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_3.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_11.__repr__()
    assert var_14 == "ValidationError(text='Position(line_no=True, column_no=True, char_index={None: True})', code='custom')"
    var_15 = var_11.__eq__(var_0)
    assert var_15 is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, position=var_15, start_position=var_0, end_position=var_10)

def test_case_38():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_9.__repr__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = var_3.__eq__(var_0)
    assert var_13 is False
    var_14 = var_2.__getitem__(var_0)
    assert var_14 is True
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_6, error=var_10)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.ValidationError(text=var_5, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_3.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult(value=var_11, error=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_13.value).__module__}.{type(var_13.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.value) == 1
    assert var_13.error is None
    var_14 = var_11.__repr__()
    assert var_14 == "ValidationError(text='Position(line_no=True, column_no=True, char_index={None: True})', code='custom')"
    var_15 = var_11.__eq__(var_0)
    assert var_15 is False
    var_16 = var_11.__eq__(var_11)
    assert var_16 is True
    var_17 = var_15.__hash__()
    assert var_17 == 0
    var_18 = var_15.__str__()
    assert var_18 == 'False'
    var_19 = var_11.items()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_19) == 1
    var_20 = var_19.__repr__()
    assert var_20 == "ItemsView(ValidationError(text='Position(line_no=True, column_no=True, char_index={None: True})', code='custom'))"
    module_0.ValidationError(key=var_18)

def test_case_40():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_7 = var_4.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Message(text=var_0, code=var_6, key=var_6, index=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'Position(line_no=True, column_no=True, char_index={None: True})'
    assert var_8.index == ['Position(line_no=True, column_no=True, char_index={None: True})']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_3.__eq__(var_0)
    assert var_11 is False
    var_12 = var_4.__repr__()
    assert var_12 == "Message(text=True, code='custom')"
    var_13 = var_10.__repr__()
    assert var_13 == 'ValidationResult(value=None)'
    var_14 = module_0.ValidationError(text=var_12, position=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_3.__eq__(var_0)
    assert var_15 is False
    var_16 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_14.__str__()
    assert var_17 == "Message(text=True, code='custom')"
    var_18 = var_8.__hash__()
    assert var_18 == 5040993273285481477
    var_19 = [var_4, var_4]
    var_20 = module_0.ValidationError(position=var_0, messages=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20) == 1
    var_21 = var_20.__repr__()
    assert var_21 == "ValidationError([Message(text=True, code='custom'), Message(text=True, code='custom')])"

def test_case_41():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == {None: True}
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = 'Rf/e'
    var_5 = []
    var_6 = module_0.Message(text=var_4, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Rf/e'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__repr__()
    assert var_7 == 'Position(line_no={None: True}, column_no=None, char_index=None)'
    var_8 = var_6.__eq__(var_0)
    assert var_8 is False
    var_9 = 'XtO<|NH-\x0c1nndqEjw'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_9, code=var_0, key=var_9, index=var_5, start_position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = '4de3BEs&jy;'
    var_7 = var_4.__eq__(var_4)
    assert var_7 is True
    var_8 = module_0.Message(text=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == {None: True}
    assert var_10.error is None
    var_11 = var_4.__repr__()
    assert var_11 == "Message(text=True, code='custom')"
    var_12 = module_0.ValidationError(text=var_6, key=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_10.__repr__()
    assert var_13 == 'ValidationResult(value={None: True})'
    var_14 = var_10.__bool__()
    assert var_14 is True
    var_15 = var_12.__eq__(var_6)
    assert var_15 is False
    var_16 = var_15.__eq__(var_0)
    var_17 = module_0.ValidationResult(error=var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert var_17.error is False
    var_18 = var_17.__repr__()
    assert var_18 == 'ValidationResult(error=False)'
    var_19 = var_15.__repr__()
    assert var_19 == 'False'
    var_20 = var_12.__eq__(var_0)
    assert var_20 is False
    var_21 = var_20.__hash__()
    assert var_21 == 0
    var_22 = var_12.__str__()
    assert var_22 == '{"Message(text=True, code=\'custom\')": \'4de3BEs&jy;\'}'
    module_0.ValidationError(code=var_20, key=var_16)

def test_case_43():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = '4de3BEs&jy;'
    var_7 = var_4.__eq__(var_4)
    assert var_7 is True
    var_8 = module_0.Message(text=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == {None: True}
    assert var_10.error is None
    var_11 = var_4.__repr__()
    assert var_11 == "Message(text=True, code='custom')"
    var_12 = module_0.ValidationError(text=var_6, key=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_10.__repr__()
    assert var_13 == 'ValidationResult(value={None: True})'
    var_14 = var_10.__bool__()
    assert var_14 is True
    var_15 = var_3.__eq__(var_0)
    assert var_15 is False
    var_16 = module_0.ValidationResult(value=var_0, error=var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert f'{type(var_16.error).__module__}.{type(var_16.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16.error) == 1
    var_17 = var_12.__repr__()
    assert var_17 == 'ValidationError([Message(text=\'4de3BEs&jy;\', code=\'custom\', index=["Message(text=True, code=\'custom\')"])])'
    var_18 = var_12.__eq__(var_0)
    assert var_18 is False
    var_19 = var_18.__hash__()
    assert var_19 == 0
    var_20 = var_12.__str__()
    assert var_20 == '{"Message(text=True, code=\'custom\')": \'4de3BEs&jy;\'}'
    var_21 = module_0.ValidationError(text=var_18, code=var_18, key=var_0)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_21) == 1
    var_22 = var_12.__repr__()
    assert var_22 == 'ValidationError([Message(text=\'4de3BEs&jy;\', code=\'custom\', index=["Message(text=True, code=\'custom\')"])])'

def test_case_44():
    var_0 = 'error_code_1'
    var_1 = 'users'
    var_2 = 0
    var_3 = [var_1, var_2, var_0]
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error_code_1'
    assert var_4.code == 'error_code_1'
    assert var_4.index == ['users', 0, 'error_code_1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error message 2'
    var_6 = 'error_code_2'
    var_7 = module_0.Message(text=var_5, code=var_6, index=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error message 2'
    assert var_7.code == 'error_code_2'
    assert var_7.index == ['users', 0, 'error_code_1']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_4, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: True}
    var_4 = module_0.Message(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: True})'
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = [var_4, var_4]
    var_8 = module_0.Message(text=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_8.text).__module__}.{type(var_8.text).__qualname__}' == 'builtins.list'
    assert len(var_8.text) == 2
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is True
    assert var_10.error is None
    var_11 = var_8.__repr__()
    assert var_11 == "Message(text=[Message(text=None, code='custom'), Message(text=None, code='custom')], code='custom')"
    var_12 = module_0.ValidationError(text=var_0, messages=var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_10.__repr__()
    assert var_13 == 'ValidationResult(value=True)'
    var_14 = var_10.__bool__()
    assert var_14 is True
    var_15 = var_3.__eq__(var_0)
    assert var_15 is False
    var_16 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_12.__repr__()
    assert var_17 == "ValidationError([Message(text=None, code='custom'), Message(text=None, code='custom')])"
    var_18 = var_12.__eq__(var_0)
    assert var_18 is False
    var_19 = var_8.__hash__()
    assert var_19 == -3870499989249068353
    var_20 = var_12.__str__()
    assert var_20 == "{'': None}"
    module_0.ValidationError(code=var_18, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = None
    var_1 = True
    var_2 = {var_0: var_0}
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index == {None: None}
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is True
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=True, column_no=True, char_index={None: None})'
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = module_0.Message(text=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = False
    var_9 = module_0.Position(var_8, var_1, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is False
    assert var_9.column_no is True
    assert var_9.char_index is None
    var_10 = var_3.__eq__(var_9)
    assert var_10 is False
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_7.__repr__()
    assert var_12 == "Message(text=None, code='custom')"
    module_0.ValidationError(position=var_0)