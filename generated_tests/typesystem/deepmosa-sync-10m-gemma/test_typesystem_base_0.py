# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 'O}Psf,!Dl"2'
    var_1 = module_0.ParseError(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1

def test_case_1():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = repr(var_1)
    assert var_2 == "Message(text='error', code='custom')"

def test_case_2():
    var_0 = 'R`\nLKP'
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0)

def test_case_3():
    var_0 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_0)

def test_case_4():
    var_0 = None
    var_1 = 'O}Ps(,!Dl"2'
    var_2 = module_0.ParseError(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_2)
    assert var_3 is False

def test_case_5():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None

def test_case_6():
    var_0 = 3
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 3
    assert var_1.column_no == 3
    assert var_1.char_index == 3
    var_2 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 3
    assert var_2.column_no == 3
    assert var_2.char_index == 3
    var_3 = bool(var_1 != var_2)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = "R{U!la+3\rsB4zs'"
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, key=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1
    var_4 = var_2.__repr__()
    assert var_4 == 'ValidationError(text="R{U!la+3\\rsB4zs\'", code=\'custom\')'
    module_0.ParseError(text=var_1, code=var_3, position=var_1)

def test_case_8():
    var_0 = 'err_code'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'err_code'
    assert var_1.code == 'err_code'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 != var_1)

def test_case_9():
    var_0 = 'u/'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == "{'u/': 'u/'}"

def test_case_10():
    var_0 = 'error'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 != 'not a message')
    assert var_2 is True

def test_case_11():
    var_0 = 'R`\nLKP'
    var_1 = None
    var_2 = False
    var_3 = module_0.Position(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is False
    assert var_3.char_index is False
    var_4 = module_0.Message(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'R`\nLKP'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = None
    var_6 = var_4.__eq__(var_5)
    assert var_6 is False
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_3.__repr__()
    assert var_9 == 'Position(line_no=None, column_no=False, char_index=False)'

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'u/'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert var_3 == 'u/'
    var_3.values()

def test_case_13():
    var_0 = None
    var_1 = 'u/'
    var_2 = None
    var_3 = module_0.ParseError(text=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_0.__repr__()
    assert var_4 == 'None'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_1, code=var_2, position=var_1, start_position=var_2, end_position=var_1)

def test_case_14():
    var_0 = 'x$8nkzYL'
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'x$8nkzYL'
    assert var_1.code == 'x$8nkzYL'
    assert var_1.index == ['x$8nkzYL']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 != var_1)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'u/'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    assert var_2 == 4130461313697664838
    var_3 = var_1.__str__()
    assert var_3 == "{'u/': 'u/'}"
    var_4 = var_1.__repr__()
    assert var_4 == "ParseError([Message(text='u/', code='custom', index=['u/'])])"
    var_0.__contains__(var_2)

def test_case_16():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__eq__(var_1)
    assert var_5 is True
    var_6 = var_1.__hash__()
    assert var_6 == 7839505873636305307

def test_case_17():
    var_0 = 'error'
    var_1 = 'username'
    var_2 = module_0.Message(text=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'error'
    assert var_2.code == 'custom'
    assert var_2.index == ['username']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = repr(var_2)
    assert var_3 == "Message(text='error', code='custom', index=['username'])"

def test_case_18():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = var_0.__bool__()
    assert var_2 is True

def test_case_19():
    var_0 = None
    var_1 = 3131
    var_2 = False
    var_3 = None
    var_4 = module_0.Position(var_1, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 3131
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = '"A$F@V<%|'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '{\'"A$F@V<%|\': \'"A$F@V<%|\'}'
    var_3 = var_1.__repr__()
    assert var_3 == 'ParseError([Message(text=\'"A$F@V<%|\', code=\'custom\', index=[\'"A$F@V<%|\'])])'
    var_4 = var_1.messages()
    var_5 = None
    var_6 = var_1.__eq__(var_5)
    assert var_6 is False
    var_7 = var_1.__len__()
    assert var_7 == 1
    var_8 = var_7.__hash__()
    assert var_8 == 1
    var_9 = var_1.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_9.__getitem__(var_6)

def test_case_21():
    var_0 = 'u/'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_2, key=var_0, index=var_1)

def test_case_22():
    var_0 = 'error'
    var_1 = 'users'
    var_2 = 0
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error'
    assert var_4.code == 'custom'
    assert var_4.index == ['users', 0]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = repr(var_4)
    assert var_5 == "Message(text='error', code='custom', index=['users', 0])"

def test_case_23():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success'
    assert var_1.error is None
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'success'
    var_4 = next(var_2)
    assert var_4 is None

def test_case_24():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.ParseError(text=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_1.__hash__()
    assert var_6 == 7839505873636305307
    var_7 = None
    var_8 = module_0.Message(text=var_1, end_position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_8.text).__module__}.{type(var_8.text).__qualname__}' == 'typesystem.base.Message'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__hash__()
    assert var_9 == 7839505873636305307
    var_10 = var_5.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    var_11 = var_1.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.Message(text=var_10, key=var_10, position=var_6, start_position=var_10)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = module_0.ParseError(text=var_3, code=var_0, position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_1)
    assert var_7 is True
    var_8 = var_6.messages()
    var_9 = var_1.__hash__()
    assert var_9 == 7839505873636305307
    var_10 = var_6.__iter__()
    var_11 = var_10.__repr__()
    module_0.ParseError(key=var_10, messages=var_10)

def test_case_26():
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
    var_7 = bool(var_6 != 'not an error object')
    assert var_7 is True
    var_8 = bool(var_6 != 123)
    assert var_8 is True

def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.ParseError(text=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_5.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = var_5.__eq__(var_8)
    assert var_9 is False
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_1, error=var_9)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = '16X*m#fP1xe\rv".{?S'
    var_2 = []
    var_3 = module_0.Message(text=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '16X*m#fP1xe\rv".{?S'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == 'Message(text=\'16X*m#fP1xe\\rv".{?S\', code=\'custom\')'
    var_5 = var_3.__repr__()
    assert var_5 == 'Message(text=\'16X*m#fP1xe\\rv".{?S\', code=\'custom\')'
    var_6 = var_3.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ParseError(text=var_1, code=var_0, position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_7.__repr__()
    assert var_8 == 'ParseError(text=\'16X*m#fP1xe\\rv".{?S\', code=\'custom\')'
    var_9 = var_7.messages(add_prefix=var_8)
    var_10 = var_3.__hash__()
    assert var_10 == 7839505873636305307
    var_11 = None
    var_12 = True
    var_13 = module_0.Position(var_11, var_11, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no is None
    assert var_13.char_index is True
    var_14 = var_7.values()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_14) == 1
    var_14.get(var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'error'
    var_1 = 'ji8E\n"TCJ(Hsv4 t\r'
    var_2 = None
    var_3 = module_0.Message(text=var_1, key=var_2, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'ji8E\n"TCJ(Hsv4 t\r'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == 'error'
    assert var_3.end_position == 'error'
    var_4 = 'user'
    var_5 = module_0.Message(text=var_0, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'error'
    assert var_5.code == 'custom'
    assert var_5.index == ['user']
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_3.__repr__()
    assert var_6 == 'Message(text=\'ji8E\\n"TCJ(Hsv4 t\\r\', code=\'custom\', position=\'error\')'
    var_7 = [var_0, var_4]
    var_8 = var_5.__repr__()
    assert var_8 == "Message(text='error', code='custom', index=['user'])"
    var_9 = module_0.Message(text=var_0, index=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'error'
    assert var_9.code == 'custom'
    assert var_9.index == ['error', 'user']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = module_0.ValidationError(text=var_5, position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_10.__contains__(var_2)
    assert var_11 is False
    var_11.get(var_0)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = 'UXb'
    var_2 = (638.430989+439.845j)
    var_3 = module_0.Message(text=var_1, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'UXb'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == (638.430989+439.845j)
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text='UXb', code='custom', start_position=(638.430989+439.845j), end_position=None)"
    var_6 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = module_0.ParseError(text=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    var_9 = var_7.messages()
    var_10 = var_3.__hash__()
    assert var_10 == 7839505873636305307
    var_11 = 1243
    var_12 = var_7.__contains__(var_0)
    assert var_12 is False
    var_13 = var_7.items()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_13) == 1
    var_14 = var_13.__repr__()
    assert var_14 == 'ItemsView(ParseError(text="Message(text=\'UXb\', code=\'custom\', start_position=(638.430989+439.845j), end_position=None)", code=\'custom\'))'
    module_0.ValidationError(text=var_12, key=var_11, messages=var_13)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = [var_1, var_1, var_1, var_1]
    var_5 = module_0.ParseError(text=var_0, code=var_0, position=var_0, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = 2171
    var_7 = module_0.Position(var_0, var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no == 2171
    assert var_7.char_index is False
    var_8 = var_1.__eq__(var_0)
    assert var_8 is False
    var_9 = var_5.messages()
    var_10 = module_0.Position(var_0, var_0, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is None
    assert var_10.char_index is False
    var_11 = var_1.__hash__()
    assert var_11 == 7839505873636305307
    var_12 = var_5.__contains__(var_2)
    assert var_12 is False
    var_13 = var_12.__repr__()
    assert var_13 == 'False'
    var_14 = var_5.__repr__()
    assert var_14 == "ParseError([Message(text=None, code='custom'), Message(text=None, code='custom'), Message(text=None, code='custom'), Message(text=None, code='custom')])"
    var_15 = var_5.items()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_15) == 1
    var_16 = var_15.__eq__(var_0)
    var_12.keys()

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.ValidationError(text=var_3, key=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.__iter__()
    var_7 = var_6.__eq__(var_0)
    var_8 = module_0.ParseError(text=var_3, code=var_0, position=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_6.__str__()
    var_10 = var_8.__iter__()
    module_0.ParseError(code=var_0, position=var_10, messages=var_10)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = module_0.ParseError(text=var_3, code=var_0, position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_6.__contains__(var_0)
    assert var_9 is False
    var_10 = var_9.__repr__()
    assert var_10 == 'False'
    module_0.ValidationError(code=var_5, messages=var_9)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.ParseError(text=var_3, code=var_0, position=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_1.__eq__(var_1)
    assert var_6 is True
    var_7 = var_5.messages()
    var_8 = var_5.__repr__()
    assert var_8 == 'ParseError(text="Message(text=None, code=\'custom\')", code=\'custom\')'
    var_9 = -1786
    var_10 = module_0.Position(var_6, var_9, var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is True
    assert var_10.column_no == -1786
    assert var_10.char_index is True
    var_11 = var_1.__hash__()
    assert var_11 == 7839505873636305307
    var_12 = var_5.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = module_0.ValidationResult(error=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert f'{type(var_13.error).__module__}.{type(var_13.error).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13.error) == 1
    var_14 = var_13.__repr__()
    assert var_14 == 'ValidationResult(error=KeysView(ParseError(text="Message(text=None, code=\'custom\')", code=\'custom\')))'
    var_15 = var_12.__str__()
    assert var_15 == 'KeysView(ParseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    var_16 = var_5.__eq__(var_12)
    assert var_16 is False
    var_17 = var_16.__repr__()
    assert var_17 == 'False'
    var_18 = module_0.ValidationError(text=var_12, code=var_14, position=var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18) == 1
    var_19 = var_12.__len__()
    assert var_19 == 1
    var_16.keys()

def test_case_35():
    var_0 = '\nCo|G<VwZnF'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '\nCo|G<VwZnF'
    assert var_1.code == '\nCo|G<VwZnF'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'error 2'
    var_3 = module_0.Message(text=var_2, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error 2'
    assert var_3.code == '\nCo|G<VwZnF'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_1 != var_3)
    assert var_4 is True

def test_case_36():
    var_0 = 'error'
    var_1 = 'code_a'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'error'
    assert var_2.code == 'code_a'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'code_b'
    var_4 = module_0.Message(text=var_0, code=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error'
    assert var_4.code == 'code_b'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = bool(var_2 != var_4)
    assert var_5 is True

def test_case_37():
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

def test_case_38():
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

def test_case_39():
    var_0 = 'error'
    var_1 = 'ji8E\n"TCJ(Hsv4 t\r'
    var_2 = None
    var_3 = module_0.Message(text=var_1, key=var_2, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'ji8E\n"TCJ(Hsv4 t\r'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == 'error'
    assert var_3.end_position == 'error'
    var_4 = module_0.Message(text=var_0, key=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error'
    assert var_4.code == 'custom'
    assert var_4.index == ['ji8E\n"TCJ(Hsv4 t\r']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_0, var_1]
    var_6 = var_4.__repr__()
    assert var_6 == 'Message(text=\'error\', code=\'custom\', index=[\'ji8E\\n"TCJ(Hsv4 t\\r\'])'
    var_7 = module_0.Message(text=var_0, index=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'error'
    assert var_7.code == 'custom'
    assert var_7.index == ['error', 'ji8E\n"TCJ(Hsv4 t\r']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = None
    var_9 = var_4.__eq__(var_8)
    assert var_9 is False
    var_10 = bool(var_4 != var_7)
    assert var_10 is True

def test_case_40():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 0
    var_3 = 'name'
    var_4 = [var_0, var_2, var_3]
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Error 1'
    assert var_5.code == 'code1'
    assert var_5.index == ['Error 1', 0, 'name']
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = [var_5, var_5]
    var_7 = module_0.BaseError(messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_7.messages()
    var_9 = len(var_8)
    assert var_9 == 2
    with pytest.raises(KeyError):
        var_10 = var_7['users'][0]['name']
    assert var_10 == 'Error 1'

def test_case_41():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 1'
    assert var_3.code == 'code1'
    assert var_3.index == ['code1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = 'key2'
    var_7 = [var_6]
    var_8 = module_0.Message(text=var_4, code=var_5, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error 2'
    assert var_8.code == 'code2'
    assert var_8.index == ['key2']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_3, var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 2
    var_11 = [var_3, var_8]
    var_12 = module_0.ValidationError(messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 2
    var_13 = bool(var_10 == var_12)
    assert var_13 is True

def test_case_42():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'code1'
    assert var_4.index == ['key1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'code2'
    assert var_9.index == ['key2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = var_11._messages
    var_13 = len(var_12)
    var_14 = bool(var_13 != 1)
    assert var_14 is True
    var_15 = str(var_11)
    var_16 = dict(var_11)
    var_17 = str(var_16)
    var_18 = bool(var_15 == var_17)
    assert var_18 is True

def test_case_43():
    var_0 = 'error'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'error'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == 'error'
    assert var_2.end_position == 'error'
    var_3 = module_0.Message(text=var_0, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='error', code='custom')"
    var_5 = None
    var_6 = var_3.__eq__(var_5)
    assert var_6 is False
    var_7 = bool(var_3 != var_2)
    assert var_7 is True