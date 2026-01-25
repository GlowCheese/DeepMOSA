# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 'kW"%j,>z8vfa'
    var_1 = True
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, position=var_1, end_position=var_0)

def test_case_1():
    var_0 = True
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0)

def test_case_2():
    var_0 = 'g4'
    var_1 = module_0.Message(text=var_0, code=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'g4'
    assert var_1.code == 'g4'
    assert var_1.index == 'g4'
    assert var_1.start_position == 'g4'
    assert var_1.end_position == 'g4'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_3():
    var_0 = ''
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ''
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_4():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_1.__len__()
    assert var_5 == 1
    var_6 = None
    var_7 = var_4.__eq__(var_6)
    assert var_7 is False

def test_case_5():
    var_0 = None
    var_1 = {}
    var_2 = -778
    var_3 = module_0.ValidationError(text=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False

def test_case_6():
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_7():
    var_0 = 42
    var_1 = 'Error message'
    var_2 = module_0.ValidationError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == 42
    assert var_3.error is None
    var_4 = list(var_3)
    var_5 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = list(var_5)

def test_case_8():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_9():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None

def test_case_10():
    var_0 = 'This is an error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'This is an error message'
    assert var_3.code == 'error_code'
    assert var_3.index == ['error_key']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='This is an error message', code='error_code', index=['error_key'])])"

def test_case_11():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_3)
    assert var_6 is False
    var_7 = None
    var_8 = var_7.__eq__(var_6)

def test_case_12():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 42
    assert var_1.error is None
    var_2 = 'error'
    var_3 = module_0.ValidationError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_1.__bool__()
    assert var_4 is True
    var_5 = module_0.ValidationResult(error=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = module_0.ValidationError(text=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_5.__repr__()
    assert var_7 == "ValidationResult(error=ValidationError(text='error', code='custom'))"

def test_case_13():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.ValidationResult()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = var_1.__repr__()
    assert var_4 == 'ValidationResult(value=None)'

def test_case_14():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = module_0.Message(text=var_0, code=var_3, start_position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '$E4'
    assert f'{type(var_5.code).__module__}.{type(var_5.code).__qualname__}' == 'typesystem.base.Position'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_5.end_position is None
    var_6 = module_0.ParseError(text=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = module_0.Position(var_2, var_0, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 3
    assert var_7.column_no == '$E4'
    assert var_7.char_index == 3
    var_8 = repr(var_3)
    var_9 = repr(var_0)
    var_10 = var_4.__eq__(var_5)
    assert var_10 is False

def test_case_15():
    var_0 = 42
    var_1 = 'Error message'
    var_2 = module_0.ValidationError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == 42
    assert var_3.error is None
    var_4 = list(var_3)
    var_5 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = list(var_2)

def test_case_16():
    var_0 = False
    var_1 = None
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is False
    assert var_2.char_index is None
    var_3 = module_0.ParseError(text=var_0, code=var_2, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_1)
    assert var_4 is False
    var_5 = var_4.__hash__()
    assert var_5 == 0
    var_6 = var_3.messages()
    var_7 = var_3.__iter__()
    var_8 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_8)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 2162
    var_1 = []
    module_0.ValidationError(key=var_0, messages=var_1)

def test_case_18():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = [var_0, var_0]
    var_6 = var_5.__len__()
    assert var_6 == 2
    var_7 = None
    var_8 = module_0.Message(text=var_0, code=var_7, start_position=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '$E4'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_8.end_position is None
    var_9 = module_0.ParseError(text=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__hash__()
    assert var_10 == -8105005508951182673
    var_11 = var_1.__eq__(var_7)
    var_12 = repr(var_9)
    var_13 = repr(var_1)
    var_14 = var_4.__eq__(var_8)
    assert var_14 is False

def test_case_19():
    var_0 = ''
    var_1 = []
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == ''
    assert var_4.code == ''
    assert var_4.index == []
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_4.__hash__()
    assert var_5 == -5190988587195632635
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = '}/]n'
    var_4 = module_0.Message(text=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '}/]n'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = '{BB,y$]ch\n'
    var_6 = module_0.Message(text=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == '{BB,y$]ch\n'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__eq__(var_2)
    assert var_7 is False
    var_8 = None
    var_9 = [var_4, var_4, var_1]
    var_10 = module_0.ValidationError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_10.__contains__(var_8)
    assert var_11 is False
    var_12 = var_4.__hash__()
    assert var_12 == -3106816870875445794
    var_13 = var_10.__iter__()
    module_0.ParseError(code=var_4, messages=var_4)

def test_case_21():
    var_0 = None
    var_1 = '}/]n'
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '}/]n'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='}/]n', code='custom')"
    var_5 = None
    var_6 = [var_2, var_2]
    var_7 = module_0.ValidationError(messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.__str__()
    assert var_8 == "{'': '}/]n'}"
    var_9 = var_7.keys()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_9) == 1
    var_10 = module_0.Position(var_5, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no == "{'': '}/]n'}"
    assert f'{type(var_10.char_index).__module__}.{type(var_10.char_index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10.char_index) == 1
    var_11 = var_8.__iter__()
    var_12 = var_9.__eq__(var_9)
    assert var_12 is True
    var_13 = var_7.__iter__()
    var_14 = var_7.__iter__()

def test_case_22():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'Error 1'
    var_5 = 'error1'
    var_6 = 'field1'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error 1'
    assert var_8.code == 'error1'
    assert var_8.index == ['field1']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = 'Error 2'
    var_10 = 'error2'
    var_11 = 'field2'
    var_12 = [var_11]
    var_13 = module_0.Message(text=var_9, code=var_10, index=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'Error 2'
    assert var_13.code == 'error2'
    assert var_13.index == ['field2']
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = [var_8, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 2
    var_16 = repr(var_15)
    var_17 = module_0.BaseError(text=var_0, code=var_1, key=var_6)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = repr(var_17)
    assert var_18 == "BaseError([Message(text='Error message', code='error_code', index=['field1'])])"

def test_case_23():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = module_0.Message(text=var_0, code=var_3, start_position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '$E4'
    assert f'{type(var_5.code).__module__}.{type(var_5.code).__qualname__}' == 'typesystem.base.Position'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_5.end_position is None
    var_6 = module_0.Position(var_2, var_0, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 3
    assert var_6.column_no == '$E4'
    assert var_6.char_index == 3
    var_7 = repr(var_2)
    var_8 = var_7.__repr__()
    assert var_8 == "'3'"
    var_9 = var_4.__eq__(var_5)
    assert var_9 is False

def test_case_24():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = [var_0, var_0]
    var_6 = var_5.__len__()
    assert var_6 == 2
    var_7 = None
    var_8 = module_0.Message(text=var_0, code=var_7, start_position=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '$E4'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_8.end_position is None
    var_9 = module_0.ParseError(text=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == '$E4'
    var_11 = repr(var_9)
    var_12 = repr(var_1)
    var_13 = var_4.__eq__(var_8)
    assert var_13 is False

def test_case_25():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 28
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 28
    assert var_3.column_no == 28
    assert var_3.char_index == 28
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_1.__len__()
    assert var_5 == 1
    var_6 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 28
    assert var_6.column_no == 28
    assert var_6.char_index == 28
    var_7 = module_0.Message(text=var_3, code=var_6, start_position=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_7.text).__module__}.{type(var_7.text).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.code).__module__}.{type(var_7.code).__qualname__}' == 'typesystem.base.Position'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_7.end_position is None
    var_8 = module_0.ParseError(text=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = repr(var_6)
    var_10 = var_9.__len__()
    assert var_10 == 49
    var_11 = repr(var_10)
    var_12 = var_4.__eq__(var_7)
    assert var_12 is False

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = '`gvi'
    var_7 = var_1.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.Message(text=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '`gvi'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_3)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_2, error=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == "Message(text=None, code='custom')"
    assert var_10.error is None
    var_11 = [var_5]
    var_12 = module_0.ValidationError(messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_12.__str__()
    assert var_13 == '}/]n'
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = module_0.Position(var_14, var_13, var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.line_no).__module__}.{type(var_15.line_no).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.column_no == '}/]n'
    assert var_15.char_index is False
    var_16 = var_12.messages(add_prefix=var_0)
    var_17 = var_15.__repr__()
    assert var_17 == 'Position(line_no=ValidationResult(value=None), column_no=}/]n, char_index=False)'
    var_18 = var_12.keys()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_18) == 1
    var_19 = var_12.messages(add_prefix=var_18)
    var_20 = var_12.__contains__(var_1)
    assert var_20 is False
    var_20.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__hash__()
    assert var_6 == -3106816870875445794
    var_7 = '`gvi'
    var_8 = module_0.Message(text=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '`gvi'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_3)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_2, error=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == "Message(text=None, code='custom')"
    assert var_10.error is None
    module_0.ValidationError(text=var_4, messages=var_7)

def test_case_28():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = '{BB,y$]ch\n'
    var_7 = module_0.Message(text=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '{BB,y$]ch\n'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_3)
    assert var_8 is False
    var_9 = module_0.ValidationResult(value=var_2, error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == "Message(text=None, code='custom')"
    assert var_9.error is None
    var_10 = None
    var_11 = module_0.Message(text=var_2, key=var_3, index=var_0, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == "Message(text=None, code='custom')"
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = [var_5]
    var_13 = var_9.__bool__()
    assert var_13 is True
    var_14 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_9.__repr__()
    assert var_15 == 'ValidationResult(value="Message(text=None, code=\'custom\')")'
    var_16 = var_14.__str__()
    assert var_16 == '}/]n'
    var_17 = var_14.__iter__()
    var_18 = var_14.__iter__()
    var_19 = var_14.__str__()
    assert var_19 == '}/]n'
    var_20 = module_0.ValidationResult()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_20.value is None
    assert var_20.error is None
    var_21 = var_14.items()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_21) == 1
    var_22 = var_21.__str__()
    assert var_22 == "ItemsView(ValidationError(text='}/]n', code='custom'))"
    var_23 = var_14.__str__()
    assert var_23 == '}/]n'
    var_24 = var_5.__eq__(var_6)
    assert var_24 is False
    var_25 = var_14.__eq__(var_14)
    assert var_25 is True
    var_26 = module_0.Position(var_8, var_10, var_8)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.Position'
    assert var_26.line_no is False
    assert var_26.column_no is None
    assert var_26.char_index is False
    var_27 = 514
    var_28 = False
    var_29 = module_0.Position(var_0, var_28, var_27)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Position'
    assert var_29.line_no is None
    assert var_29.column_no is False
    assert var_29.char_index == 514
    var_30 = var_21.__eq__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = '{BB,y$]ch\n'
    var_7 = module_0.Message(text=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '{BB,y$]ch\n'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_3)
    assert var_8 is False
    var_9 = module_0.ValidationResult(value=var_2, error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == "Message(text=None, code='custom')"
    assert var_9.error is None
    var_10 = None
    var_11 = module_0.Message(text=var_2, key=var_3, index=var_0, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == "Message(text=None, code='custom')"
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = [var_5]
    module_0.ValidationError(key=var_0, position=var_11, messages=var_12)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = '{BB,y$]ch\n'
    var_8 = module_0.Message(text=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '{BB,y$]ch\n'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_3)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_2, error=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == "Message(text=None, code='custom')"
    assert var_10.error is None
    var_11 = None
    var_12 = module_0.Message(text=var_2, key=var_3, index=var_0, end_position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == "Message(text=None, code='custom')"
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = []
    var_14 = var_10.__bool__()
    assert var_14 is True
    module_0.ValidationError(messages=var_13)

def test_case_31():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = '}/]n'
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}/]n'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = '{BB,y$]ch\n'
    var_7 = module_0.Message(text=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '{BB,y$]ch\n'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_3)
    assert var_8 is False
    var_9 = module_0.ValidationResult(value=var_2, error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == "Message(text=None, code='custom')"
    assert var_9.error is None
    var_10 = None
    var_11 = module_0.Message(text=var_2, key=var_3, index=var_0, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == "Message(text=None, code='custom')"
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = [var_5]
    var_13 = var_9.__bool__()
    assert var_13 is True
    var_14 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_9.__repr__()
    assert var_15 == 'ValidationResult(value="Message(text=None, code=\'custom\')")'
    var_16 = var_1.__hash__()
    assert var_16 == -3106816870875445794
    var_17 = module_0.ValidationResult(error=var_12)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert f'{type(var_17.error).__module__}.{type(var_17.error).__qualname__}' == 'builtins.list'
    assert len(var_17.error) == 1
    var_18 = var_14.__iter__()
    var_19 = var_14.__str__()
    assert var_19 == '}/]n'
    var_20 = var_14.keys()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_20) == 1
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_7, error=var_20)

def test_case_32():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 1
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 1
    assert var_3.char_index == 1
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = var_1.__len__()
    assert var_5 == 1
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True

def test_case_33():
    var_0 = 'example'
    var_1 = 'custom'
    var_2 = 'field'
    var_3 = [var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = module_0.Position(var_4, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 2
    assert var_7.char_index == 3
    var_8 = module_0.Message(text=var_0, code=var_1, index=var_3, position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'example'
    assert var_8.code == 'custom'
    assert var_8.index == ['field']
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = repr(var_8)
    assert var_9 == "Message(text='example', code='custom', index=['field'], position=Position(line_no=1, column_no=2, char_index=3))"
    var_10 = [var_2]
    var_11 = module_0.Position(var_4, var_5, var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no == 1
    assert var_11.column_no == 2
    assert var_11.char_index == 3
    var_12 = 4
    var_13 = 5
    var_14 = 6
    var_15 = module_0.Position(var_12, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no == 4
    assert var_15.column_no == 5
    assert var_15.char_index == 6
    var_16 = module_0.Message(text=var_0, code=var_1, index=var_10, start_position=var_11, end_position=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'example'
    assert var_16.code == 'custom'
    assert var_16.index == ['field']
    assert f'{type(var_16.start_position).__module__}.{type(var_16.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_16.end_position).__module__}.{type(var_16.end_position).__qualname__}' == 'typesystem.base.Position'
    var_17 = repr(var_16)
    assert var_17 == "Message(text='example', code='custom', index=['field'], start_position=Position(line_no=1, column_no=2, char_index=3), end_position=Position(line_no=4, column_no=5, char_index=6))"
    var_18 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text == 'example'
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert var_18.start_position is None
    assert var_18.end_position is None
    var_19 = repr(var_18)
    assert var_19 == "Message(text='example', code='custom')"

def test_case_34():
    var_0 = '$E4'
    var_1 = [var_0, var_0, var_0]
    var_2 = 18
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 18
    assert var_3.column_no == 18
    assert var_3.char_index == 18
    var_4 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '$E4'
    assert var_4.code == '$E4'
    assert var_4.index == ['$E4', '$E4', '$E4']
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.Position'
    var_5 = [var_0]
    var_6 = var_5.__len__()
    assert var_6 == 1
    var_7 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 18
    assert var_7.column_no == 18
    assert var_7.char_index == 18
    var_8 = module_0.Message(text=var_0, code=var_7, start_position=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '$E4'
    assert f'{type(var_8.code).__module__}.{type(var_8.code).__qualname__}' == 'typesystem.base.Position'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_8.end_position is None
    var_9 = module_0.ParseError(text=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = module_0.Position(var_2, var_0, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 18
    assert var_10.column_no == '$E4'
    assert var_10.char_index == 18
    var_11 = repr(var_0)
    var_12 = var_4.__repr__()
    assert var_12 == "Message(text='$E4', code='$E4', index=['$E4', '$E4', '$E4'], position=Position(line_no=18, column_no=18, char_index=18))"
    var_13 = repr(var_6)
    var_14 = var_4.__eq__(var_8)
    assert var_14 is False

def test_case_35():
    var_0 = 'example'
    var_1 = [var_0]
    var_2 = 5
    var_3 = 3
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 5
    assert var_4.column_no == 3
    assert var_4.char_index == 3
    var_5 = module_0.Message(text=var_0, code=var_0, index=var_1, position=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'example'
    assert var_5.code == 'example'
    assert var_5.index == ['example']
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = repr(var_5)
    var_7 = var_6.__repr__()
    assert var_7 == '"Message(text=\'example\', code=\'example\', index=[\'example\'], position=Position(line_no=5, column_no=3, char_index=3))"'
    var_8 = module_0.Message(text=var_7, code=var_6, start_position=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '"Message(text=\'example\', code=\'example\', index=[\'example\'], position=Position(line_no=5, column_no=3, char_index=3))"'
    assert var_8.code == "Message(text='example', code='example', index=['example'], position=Position(line_no=5, column_no=3, char_index=3))"
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_8.end_position is None
    var_9 = 6
    var_10 = module_0.Position(var_3, var_6, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 3
    assert var_10.column_no == "Message(text='example', code='example', index=['example'], position=Position(line_no=5, column_no=3, char_index=3))"
    assert var_10.char_index == 6
    var_11 = module_0.Message(text=var_0, code=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'example'
    assert f'{type(var_11.code).__module__}.{type(var_11.code).__qualname__}' == 'typesystem.base.Message'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = repr(var_11)
    var_13 = var_5.__eq__(var_8)
    assert var_13 is False
    var_14 = var_4.__repr__()
    assert var_14 == 'Position(line_no=5, column_no=3, char_index=3)'

def test_case_36():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Invalid value', code='invalid')"
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Invalid value'
    assert var_4.code == 'invalid'
    assert var_4.index == "BaseError(text='Invalid value', code='invalid')"
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Missing field'
    var_6 = 'missing'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Missing field'
    assert var_9.code == 'missing'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = repr(var_11)
    var_13 = module_0.BaseError(text=var_0, code=var_1, key=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = repr(var_13)
    var_15 = var_13.__str__()
    assert var_15 == "{'Missing field': 'Invalid value'}"
    var_16 = 1
    var_17 = 5
    var_18 = 10
    var_19 = module_0.Position(var_16, var_17, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no == 1
    assert var_19.column_no == 5
    assert var_19.char_index == 10
    var_20 = module_0.BaseError(text=var_0, code=var_1, position=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = repr(var_20)

def test_case_37():
    var_0 = 'custom'
    var_1 = [var_0]
    var_2 = 36
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 36
    assert var_3.column_no == 36
    assert var_3.char_index == 36
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, code=var_0, index=var_1, position=var_3, start_position=var_3)

def test_case_38():
    var_0 = 'example'
    var_1 = 'l$OE4['
    var_2 = [var_1]
    var_3 = 36
    var_4 = None
    var_5 = var_2.__contains__(var_4)
    assert var_5 is False
    var_6 = 3
    var_7 = module_0.Position(var_3, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 36
    assert var_7.column_no == 3
    assert var_7.char_index == 3
    var_8 = module_0.Message(text=var_0, code=var_0, index=var_2, position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'example'
    assert var_8.code == 'example'
    assert var_8.index == ['l$OE4[']
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = repr(var_8)
    var_10 = var_9.__repr__()
    assert var_10 == '"Message(text=\'example\', code=\'example\', index=[\'l$OE4[\'], position=Position(line_no=36, column_no=3, char_index=3))"'
    var_11 = [var_1]
    var_12 = module_0.Position(var_3, var_3, var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == 36
    assert var_12.column_no == 36
    assert var_12.char_index == 3
    var_13 = module_0.Message(text=var_0, code=var_10, index=var_11, start_position=var_12, end_position=var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'example'
    assert var_13.code == '"Message(text=\'example\', code=\'example\', index=[\'l$OE4[\'], position=Position(line_no=36, column_no=3, char_index=3))"'
    assert var_13.index == ['l$OE4[']
    assert f'{type(var_13.start_position).__module__}.{type(var_13.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_13.end_position).__module__}.{type(var_13.end_position).__qualname__}' == 'typesystem.base.Position'
    var_14 = repr(var_13)
    var_15 = module_0.Message(text=var_0, code=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'example'
    assert var_15.code == 'Message(text=\'example\', code=\'"Message(text=\\\'example\\\', code=\\\'example\\\', index=[\\\'l$OE4[\\\'], position=Position(line_no=36, column_no=3, char_index=3))"\', index=[\'l$OE4[\'], start_position=Position(line_no=36, column_no=36, char_index=3), end_position=Position(line_no=36, column_no=3, char_index=3))'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = repr(var_15)
    var_17 = var_8.__eq__(var_4)
    assert var_17 is False

def test_case_39():
    var_0 = '$E4'
    var_1 = [var_0]
    var_2 = 3
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 3
    assert var_3.column_no == 3
    assert var_3.char_index == 3
    var_4 = None
    var_5 = module_0.Message(text=var_0, code=var_4, index=var_1, position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '$E4'
    assert var_5.code == 'custom'
    assert var_5.index == ['$E4']
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = [var_0, var_0]
    var_7 = var_6.__len__()
    assert var_7 == 2
    var_8 = None
    var_9 = module_0.Message(text=var_0, code=var_8, start_position=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == '$E4'
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_9.end_position is None
    var_10 = module_0.ParseError(text=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.__str__()
    assert var_11 == '$E4'
    var_12 = repr(var_10)
    var_13 = repr(var_1)
    var_14 = var_5.__eq__(var_9)
    assert var_14 is False
    var_15 = var_5.__eq__(var_5)
    assert var_15 is True