# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is True
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    module_0.ValidationError(code=var_0, position=var_0)

def test_case_1():
    var_0 = 0
    var_1 = 'Error'
    var_2 = module_0.Message(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == 0
    assert var_2.end_position == 0

def test_case_2():
    var_0 = ''
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ''
    assert var_1.code == ''
    assert var_1.index == ['']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.Message'
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text='', code='', index=[''])"
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_2.__bool__()
    assert var_5 is False
    var_6 = var_2.__repr__()
    assert var_6 == "ValidationResult(error=Message(text='', code='', index=['']))"

def test_case_3():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == ''
    assert var_5.error is None
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_4.__repr__()
    assert var_8 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_9 = var_5.__repr__()
    assert var_9 == "ValidationResult(value='')"
    var_10 = var_3.__repr__()
    assert var_10 == "Message(text='', code='custom')"
    var_11 = var_3.__hash__()
    assert var_11 == -3987095495402066511

def test_case_4():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1 == var_0
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = True
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is True
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    module_0.ValidationError(position=var_0, messages=var_2)

def test_case_6():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

def test_case_7():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = True
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is True
    module_0.ValidationError(code=var_0, position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'R9a6ZO\r\x0b=\x0cxWsB4=uSc'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ParseError(text='R9a6ZO\\r\\x0b=\\x0cxWsB4=uSc', code='R9a6ZO\\r\\x0b=\\x0cxWsB4=uSc')"
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_2.__getitem__(var_1)

def test_case_10():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Message(text=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'key2'
    assert var_3.code == 'custom'
    assert var_3.index == ['key1', 'key2']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5 == var_1
    assert var_6 is False

def test_case_11():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error 1'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 2'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_1, var_3]
    var_5 = None
    var_6 = module_0.ParseError(text=var_2, code=var_2, position=var_5, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_6.__hash__()
    assert var_7 == -4038366872493721860
    var_8 = module_0.BaseError(messages=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = module_0.BaseError(messages=var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_8 == var_9
    var_11 = var_10.__bool__()
    assert var_11 is False

def test_case_12():
    var_0 = ''
    var_1 = 844
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == ''
    assert var_2.code == 'custom'
    assert var_2.index == [844]
    assert var_2.start_position == ''
    assert var_2.end_position == ''
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='', code='custom', index=[844], position='')"

def test_case_13():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = 'not an error'
    var_4 = var_1 == var_3
    assert var_4 is False
    var_5 = None
    var_6 = var_4.__eq__(var_5)

def test_case_14():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'tfr.;1=kQe-q&DP(yQn'
    var_3 = [var_0, var_2]
    var_4 = module_0.Message(text=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'key2'
    assert var_4.code == 'custom'
    assert var_4.index == ['key1', 'tfr.;1=kQe-q&DP(yQn']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_4.__eq__(var_6)
    assert var_7 is False
    var_8 = module_0.ParseError(text=var_1, position=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_6 == var_2
    assert var_9 is False

def test_case_15():
    var_0 = None
    var_1 = ''
    var_2 = [var_1]
    with pytest.raises(AssertionError):
        module_0.Message(text=var_1, code=var_0, key=var_1, index=var_2)

def test_case_16():
    var_0 = 'Error message'
    var_1 = 'error_cod'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = 'not an error'
    var_4 = var_2.__repr__()
    assert var_4 == "BaseError(text='Error message', code='error_cod')"
    var_5 = var_2 == var_3
    assert var_5 is False

def test_case_17():
    var_0 = 'code'
    var_1 = 'parent'
    var_2 = 'child'
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'parent'
    assert var_4.code == 'code'
    assert var_4.index == ['parent', 'child']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = str(var_6)
    var_8 = "{'parent': {'child': 'Nested error'}}"
    var_9 = bool(var_7 == var_8)

def test_case_18():
    var_0 = 'Error 1'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error 1'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'Error 2'
    var_3 = module_0.Message(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 2'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_1, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5 == var_5

def test_case_19():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is None
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_3, error=var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = []
    module_0.ValidationError(position=var_0, messages=var_1)

def test_case_21():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'max_length'
    var_4 = module_0.Message(text=var_0, code=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'max_length'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2 == var_4
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = ';?G#nAVYor{+8g::5'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ';?G#nAVYor{+8g::5'
    assert var_3.code == 'custom'
    assert var_3.index == [';?G#nAVYor{+8g::5']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value == ';?G#nAVYor{+8g::5'
    assert var_6.error is None
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_7.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_6.__bool__()
    assert var_9 is True
    var_10 = var_4.__repr__()
    assert var_10 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_11 = var_4.__repr__()
    assert var_11 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_12 = var_6.__repr__()
    assert var_12 == "ValidationResult(value=';?G#nAVYor{+8g::5')"
    var_13 = module_0.Message(text=var_0, key=var_8, index=var_0, start_position=var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text is None
    assert var_13.code == 'custom'
    assert var_13.index == ['ValidationResult(value=None)']
    assert f'{type(var_13.start_position).__module__}.{type(var_13.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_13.end_position is None
    var_14 = var_13.__repr__()
    assert var_14 == "Message(text=None, code='custom', index=['ValidationResult(value=None)'], start_position=Position(line_no=None, column_no=None, char_index=None), end_position=None)"
    var_15 = None
    module_0.ParseError(key=var_15)

def test_case_23():
    var_0 = -15
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == -15
    assert var_2.column_no == -15
    assert var_2.char_index == 0
    var_3 = 'ErrorG'
    var_4 = module_0.Message(text=var_3, position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'ErrorG'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_4 == var_4
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 844
    var_5 = module_0.Position(var_4, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 844
    assert var_5.column_no is None
    assert var_5.char_index == 844
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_5.__repr__()
    assert var_9 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_10 = var_8.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_8.__iter__()
    var_12 = var_3.__repr__()
    assert var_12 == "Message(text='', code='custom')"
    module_0.ParseError(text=var_12, key=var_0, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 844
    var_5 = module_0.Position(var_4, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 844
    assert var_5.column_no is None
    assert var_5.char_index == 844
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_3.__repr__()
    assert var_9 == "Message(text='', code='custom')"
    var_10 = var_5.__repr__()
    assert var_10 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_11 = var_5.__repr__()
    assert var_11 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_12 = var_3.__repr__()
    assert var_12 == "Message(text='', code='custom')"
    var_13 = var_8.__iter__()
    var_14 = None
    var_15 = [var_3, var_3, var_3, var_3]
    var_16 = module_0.ParseError(messages=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_16.messages(add_prefix=var_9)
    var_18 = var_16.keys()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_18) == 1
    var_19 = var_18.__eq__(var_14)
    var_18.__hash__()

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = 844
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text='', code='custom')"
    var_7 = module_0.Position(var_5, var_0, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 844
    assert var_7.column_no is None
    assert var_7.char_index == 844
    var_8 = var_7.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_9.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_7.__repr__()
    assert var_12 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_13 = var_9.__repr__()
    assert var_13 == 'ValidationResult(value=None)'
    var_14 = var_10.__iter__()
    var_15 = var_3.__repr__()
    assert var_15 == "Message(text='', code='custom')"
    var_16 = None
    var_17 = []
    module_0.ParseError(key=var_13, position=var_16, messages=var_17)

def test_case_27():
    var_0 = 'R9a6ZO\r\x0b=\x0cxWsB4=uSc'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ParseError(text='R9a6ZO\\r\\x0b=\\x0cxWsB4=uSc', code='R9a6ZO\\r\\x0b=\\x0cxWsB4=uSc')"
    var_4 = var_2.__eq__(var_1)
    assert var_4 is False
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_6 = var_2.__str__()
    assert var_6 == 'R9a6ZO\r\x0b=\x0cxWsB4=uSc'

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 844
    var_5 = module_0.Position(var_4, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 844
    assert var_5.column_no is None
    assert var_5.char_index == 844
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_5.__repr__()
    assert var_9 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_10 = var_5.__repr__()
    assert var_10 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_11 = var_8.__iter__()
    var_12 = var_3.__repr__()
    assert var_12 == "Message(text='', code='custom')"
    var_13 = None
    var_14 = module_0.ParseError(text=var_12, key=var_0, messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.messages()
    var_16 = var_14.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_3.__eq__(var_0)
    assert var_17 is False
    var_18 = var_3.__hash__()
    assert var_18 == -3987095495402066511
    var_19 = var_3.__hash__()
    assert var_19 == -3987095495402066511
    var_20 = [var_16]
    module_0.ParseError(text=var_0, code=var_12, position=var_16, messages=var_20)

def test_case_29():
    var_0 = None
    var_1 = ''
    var_2 = 882
    var_3 = module_0.Message(text=var_1, key=var_2, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == [882]
    assert var_3.start_position == ''
    assert var_3.end_position == ''
    var_4 = module_0.Position(var_2, var_0, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 882
    assert var_4.column_no is None
    assert var_4.char_index == 882
    var_5 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_3.__eq__(var_5)
    assert var_8 is False
    var_9 = var_7.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = var_4.__repr__()
    assert var_10 == 'Position(line_no=882, column_no=None, char_index=882)'
    var_11 = var_7.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_4.__repr__()
    assert var_12 == 'Position(line_no=882, column_no=None, char_index=882)'
    var_13 = var_7.__iter__()
    var_14 = var_4.__repr__()
    assert var_14 == 'Position(line_no=882, column_no=None, char_index=882)'
    var_15 = None
    var_16 = module_0.ParseError(text=var_14, key=var_0, messages=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_3.__repr__()
    assert var_17 == "Message(text='', code='custom', index=[882], position='')"
    var_18 = var_16.messages()
    var_19 = var_3.__eq__(var_15)
    assert var_19 is False
    var_20 = var_5.__hash__()
    assert var_20 == -3987095495402066511
    var_21 = var_16.items()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_21) == 1
    var_22 = var_21.__repr__()
    assert var_22 == "ItemsView(ParseError(text='Position(line_no=882, column_no=None, char_index=882)', code='custom'))"
    var_23 = var_21.__repr__()
    assert var_23 == "ItemsView(ParseError(text='Position(line_no=882, column_no=None, char_index=882)', code='custom'))"
    var_24 = var_7.__repr__()
    assert var_24 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 844
    var_5 = module_0.Position(var_4, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 844
    assert var_5.column_no is None
    assert var_5.char_index == 844
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_7.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = var_5.__repr__()
    assert var_10 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_11 = var_5.__repr__()
    assert var_11 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_12 = var_8.__iter__()
    var_13 = var_3.__repr__()
    assert var_13 == "Message(text='', code='custom')"
    var_14 = var_3.__repr__()
    assert var_14 == "Message(text='', code='custom')"
    var_15 = var_3.__eq__(var_0)
    assert var_15 is False
    var_16 = var_14.__hash__()
    assert var_16 == 3544831175436485192
    var_17 = [var_3]
    module_0.ParseError(code=var_0, position=var_5, messages=var_17)

def test_case_31():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__hash__()
    assert var_4 == -3987095495402066511
    var_5 = 844
    var_6 = module_0.Position(var_5, var_0, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 844
    assert var_6.column_no is None
    assert var_6.char_index == 844
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_8.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_6.__repr__()
    assert var_11 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_12 = var_9.__repr__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = var_3.__repr__()
    assert var_13 == "Message(text='', code='custom')"
    var_14 = var_9.__iter__()
    var_15 = None
    var_16 = module_0.ParseError(text=var_12, key=var_0, messages=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_3.__repr__()
    assert var_17 == "Message(text='', code='custom')"
    var_18 = var_3.__eq__(var_0)
    assert var_18 is False
    var_19 = var_17.__hash__()
    assert var_19 == 3544831175436485192
    var_20 = [var_3]
    var_21 = var_16.__repr__()
    assert var_21 == "ParseError(text='ValidationResult(value=None)', code='custom')"
    var_22 = module_0.ValidationError(text=var_17, key=var_1)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_22) == 1
    var_23 = var_22.__eq__(var_9)
    assert var_23 is False
    var_24 = module_0.ParseError(messages=var_20)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_24) == 1
    var_25 = var_22.__repr__()
    assert var_25 == 'ValidationError([Message(text="Message(text=\'\', code=\'custom\')", code=\'custom\', index=[\'\'])])'
    var_26 = var_16.items()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_26) == 1
    var_27 = var_8.__repr__()
    assert var_27 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = ''
    var_2 = []
    var_3 = module_0.Message(text=var_1, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 844
    var_5 = module_0.Message(text=var_1, key=var_4, position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == ''
    assert var_5.code == 'custom'
    assert var_5.index == [844]
    assert var_5.start_position == ''
    assert var_5.end_position == ''
    var_6 = module_0.Position(var_4, var_0, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 844
    assert var_6.column_no is None
    assert var_6.char_index == 844
    var_7 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_6.__eq__(var_0)
    assert var_8 is False
    var_9 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_9.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_6.__repr__()
    assert var_12 == 'Position(line_no=844, column_no=None, char_index=844)'
    var_13 = var_9.__repr__()
    assert var_13 == 'ValidationResult(value=None)'
    var_14 = var_3.__repr__()
    assert var_14 == "Message(text='', code='custom')"
    var_15 = var_10.__iter__()
    var_16 = var_3.__repr__()
    assert var_16 == "Message(text='', code='custom')"
    var_17 = None
    var_18 = module_0.ParseError(text=var_16, key=var_0, messages=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_18) == 1
    var_19 = var_5.__repr__()
    assert var_19 == "Message(text='', code='custom', index=[844], position='')"
    var_20 = var_18.messages()
    var_21 = var_18.__hash__()
    assert var_21 == -5818683481255692927
    var_22 = var_3.__hash__()
    assert var_22 == -3987095495402066511
    var_23 = var_3.__eq__(var_5)
    assert var_23 is False
    var_24 = var_18.__str__()
    assert var_24 == "Message(text='', code='custom')"
    module_0.ParseError(code=var_17, key=var_4)

def test_case_33():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 1'
    assert var_3.code == 'code1'
    assert var_3.index == ['field1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = 'field2'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error 2'
    assert var_7.code == 'code2'
    assert var_7.index == ['field2']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_3, var_7]
    var_9 = None
    var_10 = var_3.__eq__(var_9)
    assert var_10 is False
    var_11 = module_0.BaseError(messages=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = var_11.__repr__()
    assert var_12 == "BaseError([Message(text='Error 1', code='code1', index=['field1']), Message(text='Error 2', code='code2', index=['field2'])])"
    var_13 = str(var_11)
    var_14 = "{'field1': 'Error 1', 'field2': 'Error 2'}"
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

def test_case_34():
    var_0 = 'Error 2'
    var_1 = 'code2'
    var_2 = 'field2'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 2'
    assert var_3.code == 'code2'
    assert var_3.index == ['field2']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3, var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = str(var_5)
    var_7 = "{'field1': 'Error 1', 'field2': 'Error 2'}"
    var_8 = bool(var_6 == var_7)

def test_case_35():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 5
    assert var_2.column_no == 5
    assert var_2.char_index == 10
    var_3 = 'Error message'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_3, position=var_2, start_position=var_2)

def test_case_36():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 2
    var_4 = 10
    var_5 = module_0.Position(var_3, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 2
    assert var_5.column_no == 1
    assert var_5.char_index == 10
    var_6 = 'Error'
    var_7 = module_0.Message(text=var_6, start_position=var_2, end_position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = module_0.Message(text=var_6, start_position=var_5, end_position=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = var_7 == var_8
    assert var_9 is False

def test_case_37():
    var_0 = 0
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 0
    assert var_1.column_no == 0
    assert var_1.char_index == 0
    var_2 = 5
    var_3 = 4
    var_4 = module_0.Position(var_3, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 4
    assert var_4.column_no == 5
    assert var_4.char_index == 4
    var_5 = 'Error'
    var_6 = module_0.Message(text=var_5, start_position=var_1, end_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = module_0.Message(text=var_5, start_position=var_1, end_position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = var_6 == var_7
    assert var_8 is False

def test_case_38():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 5
    var_4 = 4
    var_5 = module_0.Position(var_0, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 5
    assert var_5.char_index == 4
    var_6 = 'Error'
    var_7 = module_0.Message(text=var_6, start_position=var_2, end_position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = module_0.Message(text=var_6, start_position=var_2, end_position=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = var_7 == var_8
    assert var_9 is False

def test_case_39():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 5
    assert var_3.char_index == 10
    var_4 = 'Error message'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_4, position=var_3, end_position=var_3)

def test_case_40():
    var_0 = '?or 8essam`'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1 == var_1