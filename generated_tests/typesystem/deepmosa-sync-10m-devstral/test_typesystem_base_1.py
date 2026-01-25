# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'sI0EjqII'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None

def test_case_1():
    var_0 = 'Invalid value'
    var_1 = 'field'
    var_2 = module_0.BaseError(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)

def test_case_2():
    var_0 = 2
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 2
    assert var_1.column_no == 2
    assert var_1.char_index == 2
    var_2 = 'error_code'
    var_3 = module_0.Message(text=var_2, code=var_2, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error_code'
    assert var_3.code == 'error_code'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.Position'
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='error_code', code='error_code', position=Position(line_no=2, column_no=2, char_index=2))"

def test_case_3():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 1'
    assert var_3.code == 'code1'
    assert var_3.index == ['key1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'Error 1'
    assert var_4.index == ['Error 1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_3, var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 2
    var_7 = var_6._messages
    var_8 = bool(var_6._messages == var_5)
    assert var_8 is True
    var_9 = var_6._message_dict

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_5():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ValidationError(text='', code='custom')"
    var_4 = var_2.__len__()
    assert var_4 == 1
    var_5 = None
    var_6 = 'vTP"RWw5d~Sp:X'
    var_7 = module_0.Message(text=var_6, code=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'vTP"RWw5d~Sp:X'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'sI0EjqII'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_1.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    module_0.ParseError()

def test_case_8():
    var_0 = 'h 2\nY `3o0JNw6b'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = 'sI0EjqII'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False
    var_3 = var_1.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_3.messages(add_prefix=var_2)

def test_case_10():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'test'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 6322207303366041389
    var_4 = 'field1'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Error'
    assert var_5.code == 'test'
    assert var_5.index == ['field1']
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = 'field2'
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'test'
    assert var_7.index == ['field2']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = bool(not var_5 == var_7)
    assert var_8 is True

def test_case_11():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = var_0.__bool__()
    assert var_4 is True
    var_5 = module_0.Position(var_1, var_2, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 2
    assert var_5.char_index == 3
    var_6 = 'Error message'
    var_7 = 'error_code'
    var_8 = module_0.BaseError(text=var_6, code=var_7, position=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = module_0.Message(text=var_6, code=var_7, position=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error message'
    assert var_9.code == 'error_code'
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.end_position).__module__}.{type(var_9.end_position).__qualname__}' == 'typesystem.base.Position'
    var_10 = var_8.__repr__()
    assert var_10 == "BaseError(text='Error message', code='error_code')"
    var_11 = [var_9]
    var_12 = bool(var_8._messages == var_11)
    assert var_12 is True
    var_13 = var_8._message_dict
    var_14 = var_12.__repr__()
    assert var_14 == 'True'
    var_15 = bool(var_8._message_dict == {'': 'Error message'})
    assert var_15 is True

def test_case_12():
    var_0 = 2
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 2
    assert var_1.column_no == 2
    assert var_1.char_index == 2
    var_2 = 'Error messBge'
    var_3 = '4/I\tJ'
    var_4 = module_0.BaseError(text=var_2, code=var_3, position=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "BaseError(text='Error messBge', code='4/I\\tJ')"
    var_6 = var_4.__str__()
    assert var_6 == 'Error messBge'

def test_case_13():
    var_0 = 0
    var_1 = 2
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 0
    assert var_2.column_no == 2
    assert var_2.char_index == 0
    var_3 = 'rr_c7d\n'
    var_4 = module_0.BaseError(text=var_3, code=var_3, position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = None
    var_6 = var_2.__eq__(var_5)
    assert var_6 is False
    var_7 = var_4.__str__()
    assert var_7 == 'rr_c7d\n'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    module_0.ParseError(text=var_0, code=var_0)

def test_case_15():
    var_0 = 'sI0EjqII'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_2) == 1
    var_3 = var_2.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = var_0.__repr__()
    assert var_5 == "'sI0EjqII'"

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = '@~?WeA4_Nk'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_3.__repr__()
    assert var_4 == 'True'
    var_5 = None
    module_0.ValidationError(key=var_5, messages=var_5)

def test_case_17():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__hash__()
    assert var_3 == 6322207303366041389
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = None
    var_6 = module_0.ValidationResult(value=var_5, error=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__iter__()
    var_8 = var_6.__repr__()
    assert var_8 == 'ValidationResult(value=None)'

def test_case_18():
    var_0 = 'Error'
    var_1 = 'est'
    var_2 = module_0.Message(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'est'
    assert var_2.index == ['est']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'est'
    assert var_3.index == ['est']
    assert var_3.start_position is None
    assert var_3.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = 'Error message'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error message'
    assert var_2.code == 'custom'
    assert var_2.index == ['username']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='Error message', code='custom', index=['username'])"
    var_4 = var_2.text
    assert var_4 == 'Error message'
    var_5 = -1777
    module_0.ValidationError(key=var_3, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_3, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__hash__()
    assert var_6 == 6322207303366041389
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.ValidationError(text=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_9.__eq__(var_3)
    var_11 = var_2.get(var_1)
    var_12 = var_9.__iter__()
    var_13 = var_9.__hash__()
    var_14 = module_0.ParseError(text=var_10, position=var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_2.messages(add_prefix=var_7)
    var_16 = var_12.__repr__()
    var_9.__len__()

def test_case_21():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Invalid value'
    assert var_4.code == 'invalid'
    assert var_4.index == ['field1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Missing value'
    var_6 = 'missing'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Missing value'
    assert var_9.code == 'missing'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = str(var_11)
    assert var_12 == "{'field1': 'Invalid value', 'field2': 'Missing value'}"

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_3, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_2.messages()
    var_7 = var_4.__eq__(var_1)
    assert var_7 is False
    var_8 = None
    module_0.ValidationError(key=var_8, messages=var_8)

def test_case_23():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error'
    assert var_1.code == 'Error'
    assert var_1.index == ['Error']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 == var_1)
    assert var_2 is True

def test_case_24():
    var_0 = 1
    var_1 = 2
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 2
    assert var_2.char_index == 2
    var_3 = 'Error message'
    var_4 = 'error_code'
    var_5 = module_0.BaseError(text=var_3, code=var_4, position=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = module_0.Message(text=var_3, code=var_4, position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error message'
    assert var_6.code == 'error_code'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = None
    var_8 = var_2.__eq__(var_7)
    assert var_8 is False
    var_9 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == 1
    assert var_9.error is None
    var_10 = var_6.__repr__()
    assert var_10 == "Message(text='Error message', code='error_code', position=Position(line_no=1, column_no=2, char_index=2))"
    var_11 = var_5.__repr__()
    assert var_11 == "BaseError(text='Error message', code='error_code')"
    var_12 = [var_6]
    var_13 = var_5.__str__()
    assert var_13 == 'Error message'
    var_14 = bool(var_5._messages == var_12)
    assert var_14 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_3, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.get(var_3)
    var_6 = var_4.__eq__(var_3)
    assert var_6 is False
    var_7 = var_4.__hash__()
    assert var_7 == 6322207303366041389
    var_8 = var_4.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_4.__eq__(var_1)
    assert var_9 is False
    var_10 = module_0.ValidationError(text=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_11.__eq__(var_3)
    var_13 = '|d^'
    module_0.ParseError(code=var_13, key=var_0, messages=var_12)

def test_case_26():
    var_0 = 1
    var_1 = 3
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 3
    assert var_2.char_index == 3
    var_3 = 'error_code'
    var_4 = module_0.BaseError(text=var_3, code=var_3, position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = None
    var_6 = None
    var_7 = module_0.Message(text=var_5, code=var_6, position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = var_4.__repr__()
    assert var_8 == "BaseError(text='error_code', code='error_code')"
    var_9 = [var_7]
    var_10 = bool(var_4._messages == var_9)
    var_11 = var_4._message_dict
    var_12 = bool(var_4._message_dict == {'': 'Error message'})

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = ''
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = None
    var_4 = module_0.Message(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == ''
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__hash__()
    assert var_6 == 6322207303366041389
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text='', code='custom')"
    var_8 = var_4.__eq__(var_4)
    assert var_8 is True
    var_9 = None
    var_10 = module_0.ValidationError(text=var_1, code=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = None
    var_12 = module_0.ValidationResult(value=var_11, error=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error == ''
    var_13 = var_10.__eq__(var_9)
    assert var_13 is False
    var_14 = var_1.get(var_2)
    var_15 = var_1.items()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_15) == 1
    var_16 = var_12.__iter__()
    var_17 = var_15.__repr__()
    assert var_17 == "ItemsView(ValidationError([Message(text='', code='custom', index=[''], position='')]))"
    var_18 = var_1.__eq__(var_2)
    assert var_18 is False
    var_19 = module_0.Message(text=var_2, key=var_9, index=var_9, start_position=var_13, end_position=var_2)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is None
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is False
    assert var_19.end_position is None
    var_20 = var_12.__repr__()
    assert var_20 == "ValidationResult(error='')"
    var_21 = '>aU9y^m*\\wkk]+Y'
    var_22 = module_0.Message(text=var_21, code=var_14, start_position=var_3, end_position=var_15)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Message'
    assert var_22.text == '>aU9y^m*\\wkk]+Y'
    assert var_22.code == 'custom'
    assert var_22.index == []
    assert var_22.start_position is None
    assert f'{type(var_22.end_position).__module__}.{type(var_22.end_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_22.end_position) == 1
    var_23 = var_10.__len__()
    assert var_23 == 1
    var_24 = 'o(d\\{@`unE\rGNu'
    var_25 = [var_14]
    var_26 = module_0.Message(text=var_24, index=var_25, end_position=var_13)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.Message'
    assert var_26.text == 'o(d\\{@`unE\rGNu'
    assert var_26.code == 'custom'
    assert var_26.index == [None]
    assert var_26.start_position is None
    assert var_26.end_position is False
    module_0.ValidationError(messages=var_15)

def test_case_28():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_3.__hash__()
    assert var_5 == 6322207303366041389
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_3.__eq__(var_1)
    assert var_7 is False
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_3, error=var_2)

def test_case_29():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_3.__eq__(var_1)
    assert var_7 is False
    var_8 = module_0.ValidationError(text=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_9.__eq__(var_1)
    var_11 = var_2.get(var_1)
    var_12 = var_2.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12) == 1
    var_13 = var_9.__iter__()
    var_14 = var_2.__eq__(var_11)
    assert var_14 is False
    var_15 = var_12.__repr__()
    assert var_15 == "ItemsView(ValidationError(text='', code='custom'))"
    var_16 = var_11.__eq__(var_1)
    assert var_16 is True
    with pytest.raises(AssertionError):
        module_0.Message(text=var_6, position=var_14, start_position=var_12)

def test_case_30():
    var_0 = ''
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_3.__eq__(var_1)
    assert var_7 is False
    var_8 = module_0.ValidationError(text=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = module_1.BaseException()
    var_11 = var_2.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = var_9.__iter__()
    var_13 = var_2.__eq__(var_2)
    assert var_13 is True
    var_14 = var_11.__repr__()
    assert var_14 == "ItemsView(ValidationError(text='', code='custom'))"
    var_15 = var_5.__repr__()
    assert var_15 == '1'
    var_16 = var_11.__eq__(var_5)
    var_17 = module_0.Message(text=var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_17.text).__module__}.{type(var_17.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17.text) == 1
    assert var_17.code == 'custom'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = var_2.__len__()
    assert var_18 == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_10, key=var_11, index=var_18)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = ''
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.Message(text=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = var_3.__hash__()
    assert var_5 == 6322207303366041389
    var_6 = var_3.__eq__(var_3)
    assert var_6 is True
    var_7 = None
    var_8 = module_0.ValidationError(text=var_1, code=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = module_0.ValidationResult(value=var_7, error=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_1.__str__()
    assert var_10 == "{'': ''}"
    var_11 = var_1.get(var_2)
    var_12 = var_1.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12) == 1
    var_13 = var_9.__iter__()
    var_14 = var_12.__repr__()
    assert var_14 == "ItemsView(ValidationError([Message(text='', code='custom', index=[''], position='')]))"
    var_15 = var_1.__eq__(var_12)
    assert var_15 is False
    var_16 = module_0.Message(text=var_7, key=var_7, index=var_7, start_position=var_10, end_position=var_7)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position == "{'': ''}"
    assert var_16.end_position is None
    var_17 = var_9.__repr__()
    assert var_17 == 'ValidationResult(value=None)'
    var_18 = var_10.__len__()
    assert var_18 == 8
    var_19 = module_0.Message(text=var_2, key=var_12, start_position=var_7, end_position=var_10)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is None
    assert var_19.code == 'custom'
    assert f'{type(var_19.index).__module__}.{type(var_19.index).__qualname__}' == 'builtins.list'
    assert len(var_19.index) == 1
    assert var_19.start_position is None
    assert var_19.end_position == "{'': ''}"
    var_20 = var_11.__hash__()
    assert var_20 == 8592037097112
    var_21 = var_16.__hash__()
    assert var_21 == 6322207303366041389
    var_18.__contains__(var_18)

def test_case_32():
    var_0 = ''
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.Message(text=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ''
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = var_3.__hash__()
    assert var_5 == 6322207303366041389
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_0)

def test_case_33():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 1'
    assert var_6.code == 'code1'
    assert var_6.index == ['users', 0, 'name']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = module_0.Message(text=var_7, code=var_8, index=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'code2'
    assert var_9.index == ['users', 0, 'name']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_6, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = bool(var_11._messages == var_10)
    assert var_12 is True
    var_13 = var_11._message_dict
    var_14 = bool(var_11._message_dict == {'users': {0: {'name': 'Error 1'}, 1: {'email': 'Error 2'}}})

def test_case_34():
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
    var_5 = bool(not var_3 == var_4)
    assert var_5 is True

def test_case_35():
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
    var_5 = bool(not var_3 == var_4)
    assert var_5 is True

def test_case_36():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_3, messages=var_1)

def test_case_37():
    var_0 = 'Error'
    var_1 = 'test1'
    var_2 = 'field'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'test1'
    assert var_3.index == ['field']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'test2'
    var_5 = module_0.Message(text=var_0, code=var_4, key=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Error'
    assert var_5.code == 'test2'
    assert var_5.index == ['field']
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_38():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = module_0.Message(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'test'
    assert var_2.index == ['test']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'field2'
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'test'
    assert var_4.index == ['field2']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

def test_case_39():
    var_0 = 1
    var_1 = 3
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 3
    assert var_2.char_index == 3
    var_3 = 'Error message'
    var_4 = '40\t'
    var_5 = module_0.BaseError(text=var_3, code=var_4, position=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = module_0.Message(text=var_3, code=var_4, position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error message'
    assert var_6.code == '40\t'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = var_6.__repr__()
    assert var_7 == "Message(text='Error message', code='40\\t', position=Position(line_no=1, column_no=3, char_index=3))"
    var_8 = var_5.__len__()
    assert var_8 == 1
    var_9 = [var_6]
    var_10 = var_2.__repr__()
    assert var_10 == 'Position(line_no=1, column_no=3, char_index=3)'
    var_11 = bool(var_5._messages == var_9)
    assert var_11 is True
    with pytest.raises(AssertionError):
        module_0.Message(text=var_8, position=var_9, end_position=var_11)

def test_case_40():
    var_0 = 1
    var_1 = 2
    var_2 = (var_0, var_1)
    var_3 = 4
    var_4 = (var_3, var_3)
    var_5 = 'Error'
    var_6 = 'test'
    var_7 = module_0.Message(text=var_5, code=var_6, position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'test'
    assert var_7.index == []
    assert var_7.start_position == (1, 2)
    assert var_7.end_position == (1, 2)
    var_8 = module_0.Message(text=var_5, code=var_6, position=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'test'
    assert var_8.index == []
    assert var_8.start_position == (4, 4)
    assert var_8.end_position == (4, 4)
    var_9 = bool(not var_7 == var_8)
    assert var_9 is True

def test_case_41():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Invalid value'
    assert var_4.code == 'invalid'
    assert var_4.index == ['field1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Missing value'
    var_6 = 'missing'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Missing value'
    assert var_9.code == 'missing'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = repr(var_11)

def test_case_42():
    var_0 = 2
    var_1 = 'error_code'
    var_2 = None
    var_3 = module_0.Message(text=var_1, position=var_2, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error_code'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == 2
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='error_code', code='custom', start_position=2, end_position=None)"

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = ''
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = None
    var_4 = module_0.Message(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == ''
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_1.__str__()
    assert var_6 == "{'': ''}"
    var_7 = var_4.__hash__()
    assert var_7 == 6322207303366041389
    var_8 = var_4.__repr__()
    assert var_8 == "Message(text='', code='custom')"
    var_9 = var_4.__eq__(var_4)
    assert var_9 is True
    var_10 = None
    var_11 = module_0.ValidationError(text=var_1, code=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = module_0.ValidationResult(value=var_2, error=var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = var_11.__eq__(var_10)
    assert var_13 is False
    var_14 = var_1.get(var_2)
    var_15 = var_1.items()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_15) == 1
    var_16 = var_12.__iter__()
    var_17 = var_15.__repr__()
    assert var_17 == "ItemsView(ValidationError([Message(text='', code='custom', index=[''], position='')]))"
    var_18 = var_1.__eq__(var_2)
    assert var_18 is False
    var_19 = module_0.Message(text=var_2, key=var_10, index=var_10, start_position=var_13, end_position=var_2)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is None
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is False
    assert var_19.end_position is None
    var_20 = var_12.__repr__()
    assert var_20 == 'ValidationResult(value=None)'
    var_21 = '>aU9y^m*\\wkk]+Y'
    var_22 = module_0.Message(text=var_21, code=var_14, start_position=var_3, end_position=var_15)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Message'
    assert var_22.text == '>aU9y^m*\\wkk]+Y'
    assert var_22.code == 'custom'
    assert var_22.index == []
    assert var_22.start_position is None
    assert f'{type(var_22.end_position).__module__}.{type(var_22.end_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_22.end_position) == 1
    var_23 = var_15.__contains__(var_16)
    assert var_23 is False
    var_24 = var_11.__len__()
    assert var_24 == 1
    var_25 = 'o(d\\{@`unE\rGNu'
    var_26 = [var_14]
    var_27 = module_0.Message(text=var_25, index=var_26, end_position=var_13)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Message'
    assert var_27.text == 'o(d\\{@`unE\rGNu'
    assert var_27.code == 'custom'
    assert var_27.index == [None]
    assert var_27.start_position is None
    assert var_27.end_position is False
    module_0.ValidationError(messages=var_15)