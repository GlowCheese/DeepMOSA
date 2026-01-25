# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.ParseError(messages=var_0)

def test_case_3():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

def test_case_5():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = repr(var_3)
    assert var_4 == "ValidationResult(error=ValidationError(text='Error message', code='error_code'))"
    var_5 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == 'error_code'
    assert var_5.error is None
    var_6 = repr(var_5)

def test_case_6():
    var_0 = ' BxjGfb'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_7():
    var_0 = None
    var_1 = 514
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == 514
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = True
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is True
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    module_0.ParseError()

def test_case_9():
    var_0 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'NF?Lp54Vo0,J2\\'
    var_1 = None
    var_2 = []
    module_0.ParseError(text=var_0, code=var_1, messages=var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    module_0.ParseError(key=var_3)

def test_case_12():
    var_0 = "'BeJ"
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == "'BeJ"
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == "'BeJ"
    assert var_1.end_position == "'BeJ"
    var_2 = var_1.__repr__()
    assert var_2 == 'Message(text="\'BeJ", code=\'custom\', position="\'BeJ")'

def test_case_13():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()

def test_case_14():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = True
    var_4 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is True
    assert var_4.char_index is None
    var_5 = None
    var_6 = var_1.__eq__(var_0)
    assert var_6 is False
    var_7 = var_4.__eq__(var_5)
    assert var_7 is False
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.ValidationError(text=var_8, code=var_0, key=var_6, messages=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_9.get(var_6)
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_10.__repr__()
    assert var_11 == '"Message(text=None, code=\'custom\')"'

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = None
    var_4 = 'y\x0c}\x0b4pDq.#\r%'
    module_0.ParseError(position=var_3, messages=var_4)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = True
    var_4 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is True
    assert var_4.char_index is None
    var_5 = None
    var_6 = var_1.__eq__(var_0)
    assert var_6 is False
    var_7 = var_4.__eq__(var_5)
    assert var_7 is False
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = None
    var_10 = None
    var_11 = module_0.ValidationError(text=var_8, code=var_0, key=var_6, messages=var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_11.__len__()
    assert var_12 == 1
    var_13 = var_12.__repr__()
    assert var_13 == '1'
    module_0.ParseError(text=var_9, key=var_10, messages=var_5)

def test_case_17():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = var_0.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_18():
    var_0 = "y7~`wy\rtn4'lA]Qoog"
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0, position=var_1)

def test_case_19():
    var_0 = None
    var_1 = 'H?&v\n6e\\/c'
    var_2 = module_0.Message(text=var_1, key=var_1, start_position=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'H?&v\n6e\\/c'
    assert var_2.code == 'custom'
    assert var_2.index == ['H?&v\n6e\\/c']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='H?&v\\n6e\\\\/c', code='custom', index=['H?&v\\n6e\\\\/c'])"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text='H?&v\\n6e\\\\/c', code='custom', index=['H?&v\\n6e\\\\/c'])"

def test_case_20():
    var_0 = None
    var_1 = ' BxjGfb'
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is False
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    module_0.ParseError(key=var_1)

def test_case_22():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is None
    assert var_6.column_no is False
    assert var_6.char_index is None
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.ParseError(text=var_5, key=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = True
    var_4 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is True
    assert var_4.char_index is None
    var_5 = None
    var_6 = var_1.__eq__(var_0)
    assert var_6 is False
    var_7 = var_4.__eq__(var_5)
    assert var_7 is False
    var_8 = var_1.__eq__(var_1)
    assert var_8 is True
    var_9 = var_4.__repr__()
    assert var_9 == 'Position(line_no=None, column_no=True, char_index=None)'
    var_10 = None
    var_11 = None
    module_0.ParseError(text=var_10, key=var_11, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = '[m[(bB@v>'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '[m[(bB@v>'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '[m[(bB@v>'
    assert var_2.code == 'custom'
    assert var_2.index == '[m[(bB@v>'
    assert var_2.start_position is None
    assert var_2.end_position == '[m[(bB@v>'
    var_3 = var_2.__hash__()
    assert var_3 == -4387984856659125393
    var_4 = None
    var_5 = var_2.__eq__(var_4)
    assert var_5 is False
    var_6 = {var_2: var_2}
    var_7 = module_0.ValidationResult(error=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'builtins.dict'
    assert len(var_7.error) == 1
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    module_0.ValidationError(position=var_1, messages=var_1)

def test_case_26():
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
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.Position(var_4, var_0, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is False
    var_8 = var_5.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_5.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = [var_1, var_1]
    var_11 = module_0.ParseError(text=var_3, position=var_3, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_11.values()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_12) == 1
    var_13 = var_12.__repr__()
    assert var_13 == "ValuesView(ParseError([Message(text=None, code='custom'), Message(text=None, code='custom')]))"

def test_case_27():
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
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = module_0.Position(var_4, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_8.__eq__(var_3)
    assert var_9 is False
    var_10 = var_5.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_1.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = [var_1, var_1]
    var_13 = module_0.ParseError(text=var_3, position=var_3, messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.__str__()
    assert var_14 == "{'': None}"

def test_case_28():
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
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = module_0.Position(var_4, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_8.__eq__(var_3)
    assert var_9 is False
    var_10 = var_5.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_5.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = [var_1, var_1]
    var_13 = module_0.ParseError(text=var_3, position=var_3, messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.values()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_14) == 1
    var_15 = var_13.messages(add_prefix=var_9)

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__bool__()
    assert var_6 is True
    var_7 = module_0.Position(var_3, var_0, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is True
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = [var_1]
    var_10 = module_0.ParseError(text=var_0, position=var_0, messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = module_0.Message(text=var_6, code=var_8, key=var_6, start_position=var_7, end_position=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is True
    assert var_11.code == 'ValidationResult(value=None)'
    assert var_11.index == [True]
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_11.end_position is None
    var_12 = var_10.__str__()
    var_13 = var_11.__repr__()
    assert var_13 == "Message(text=True, code='ValidationResult(value=None)', index=[True], start_position=Position(line_no=True, column_no=None, char_index=True), end_position=None)"
    var_14 = var_10.__repr__()
    assert var_14 == "ParseError(text=None, code='custom')"
    var_15 = var_10.values()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_15) == 1
    var_16 = var_4.__repr__()
    assert var_16 == 'ValidationResult(value=None)'
    var_17 = var_10.__eq__(var_13)
    assert var_17 is False
    module_0.ParseError(code=var_15)

def test_case_30():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_0, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_6.__bool__()
    assert var_8 is True
    var_9 = module_0.Position(var_5, var_0, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is True
    assert var_9.column_no is None
    assert var_9.char_index is True
    var_10 = var_9.__eq__(var_4)
    assert var_10 is False
    var_11 = var_6.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_6.__repr__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = [var_2, var_2]
    var_14 = module_0.ParseError(text=var_4, position=var_4, messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.__hash__()
    assert var_15 == 182812365214098400
    var_16 = var_14.__str__()
    assert var_16 == "{'': None}"
    var_17 = var_6.__repr__()
    assert var_17 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.Position(var_2, var_0, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert var_5.column_no is None
    assert var_5.char_index is True
    var_6 = var_3.__repr__()
    assert var_6 == 'ValidationResult(value=None)'
    var_7 = [var_1]
    var_8 = module_0.ParseError(text=var_0, position=var_0, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = ''
    var_10 = module_0.Message(text=var_9, code=var_6, key=var_9, start_position=var_5, end_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == ''
    assert var_10.code == 'ValidationResult(value=None)'
    assert var_10.index == ['']
    assert f'{type(var_10.start_position).__module__}.{type(var_10.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_10.end_position is None
    var_11 = var_8.__str__()
    var_12 = var_1.__repr__()
    assert var_12 == "Message(text=None, code='custom')"
    var_13 = var_8.__repr__()
    assert var_13 == "ParseError(text=None, code='custom')"
    var_14 = var_8.values()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_14) == 1
    var_15 = var_14.__repr__()
    assert var_15 == "ValuesView(ParseError(text=None, code='custom'))"
    var_16 = var_8.__eq__(var_0)
    assert var_16 is False
    module_0.ParseError()

def test_case_32():
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
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_4.__hash__()
    assert var_5 == 1
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = [var_1]
    var_8 = module_0.ParseError(text=var_3, position=var_3, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.values()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_9) == 1
    var_10 = var_9.__repr__()
    assert var_10 == "ValuesView(ParseError(text=None, code='custom'))"
    var_11 = module_0.BaseError(text=var_9, key=var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_11.__str__()
    assert var_12 == "{True: ValuesView(ParseError(text=None, code='custom'))}"
    var_13 = var_6.__repr__()
    assert var_13 == 'ValidationResult(value=None)'
    var_14 = var_8.__eq__(var_3)
    assert var_14 is False

def test_case_33():
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
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = module_0.Position(var_4, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_5.__bool__()
    assert var_9 is True
    var_10 = var_5.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_5.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = [var_1, var_1]
    var_13 = module_0.ParseError(text=var_3, position=var_3, messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = ''
    var_15 = "S6\r0'UXZ4\t"
    var_16 = module_0.Message(text=var_14, code=var_10, key=var_15, start_position=var_8, end_position=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == ''
    assert var_16.code == 'ValidationResult(value=None)'
    assert var_16.index == ["S6\r0'UXZ4\t"]
    assert f'{type(var_16.start_position).__module__}.{type(var_16.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_16.end_position is None
    var_17 = var_13.__str__()
    assert var_17 == "{'': None}"
    var_18 = var_13.values()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_18) == 1
    var_19 = var_5.__repr__()
    assert var_19 == 'ValidationResult(value=None)'
    var_20 = module_0.BaseError(text=var_18)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = var_20.messages()

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__bool__()
    assert var_6 is True
    var_7 = module_0.Position(var_6, var_0, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is True
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = [var_1]
    var_10 = module_0.ParseError(text=var_0, position=var_0, messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = 'c$Yc^\r%'
    var_12 = module_0.Message(text=var_11, key=var_2, position=var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'c$Yc^\r%'
    assert var_12.code == 'custom'
    assert var_12.index == [1817854750117161950]
    assert f'{type(var_12.start_position).__module__}.{type(var_12.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_12.end_position).__module__}.{type(var_12.end_position).__qualname__}' == 'typesystem.base.Position'
    var_13 = var_10.__str__()
    var_14 = var_12.__repr__()
    assert var_14 == "Message(text='c$Yc^\\r%', code='custom', index=[1817854750117161950], position=Position(line_no=True, column_no=None, char_index=True))"
    var_15 = var_10.__repr__()
    assert var_15 == "ParseError(text=None, code='custom')"
    var_16 = var_10.get(var_10)
    var_17 = var_16.__eq__(var_4)
    var_18 = var_10.__len__()
    assert var_18 == 1
    module_0.ParseError(code=var_16, key=var_17, position=var_18, messages=var_18)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__bool__()
    assert var_6 is True
    var_7 = module_0.Position(var_3, var_0, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is True
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = [var_1]
    var_10 = module_0.ParseError(text=var_0, position=var_0, messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = ''
    var_12 = 'c$Yc^\r%'
    var_13 = module_0.Message(text=var_11, code=var_8, key=var_12, start_position=var_7, end_position=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == ''
    assert var_13.code == 'ValidationResult(value=None)'
    assert var_13.index == ['c$Yc^\r%']
    assert f'{type(var_13.start_position).__module__}.{type(var_13.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_13.end_position is None
    var_14 = var_10.__str__()
    var_15 = var_13.__repr__()
    assert var_15 == "Message(text='', code='ValidationResult(value=None)', index=['c$Yc^\\r%'], start_position=Position(line_no=True, column_no=None, char_index=True), end_position=None)"
    var_16 = var_10.__repr__()
    assert var_16 == "ParseError(text=None, code='custom')"
    var_17 = var_4.__repr__()
    assert var_17 == 'ValidationResult(value=None)'
    var_18 = var_13.__eq__(var_13)
    assert var_18 is True
    module_0.ParseError(text=var_0, code=var_15, key=var_0, messages=var_18)

def test_case_36():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = module_0.Position(var_4, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_5.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = [var_1]
    var_11 = module_0.ParseError(text=var_0, position=var_0, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = ''
    var_13 = module_0.BaseError(text=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = module_0.Message(text=var_12, code=var_9, key=var_12, start_position=var_8, end_position=var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == ''
    assert var_14.code == 'ValidationResult(value=None)'
    assert var_14.index == ['']
    assert f'{type(var_14.start_position).__module__}.{type(var_14.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_14.end_position is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_3, key=var_6, position=var_8, end_position=var_8)

def test_case_37():
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
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_4.__bool__()
    assert var_7 is True
    var_8 = module_0.Position(var_3, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = var_4.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_4.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = [var_1, var_1]
    var_13 = module_0.ParseError(text=var_0, position=var_0, messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text is None
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = var_14.__repr__()
    assert var_15 == "Message(text=None, code='custom')"
    var_16 = var_13.values()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_16) == 1
    var_17 = var_16.__repr__()
    assert var_17 == "ValuesView(ParseError([Message(text=None, code='custom'), Message(text=None, code='custom')]))"
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_7, error=var_16)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1817854750117161950
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_5.__repr__()
    assert var_6 == 'ValidationResult(value=None)'
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = module_0.Position(var_4, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_5.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = [var_1]
    var_11 = module_0.ParseError(text=var_0, position=var_0, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = 'c$Yc^\r%'
    var_13 = module_0.Message(text=var_12, key=var_2, position=var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'c$Yc^\r%'
    assert var_13.code == 'custom'
    assert var_13.index == [1817854750117161950]
    assert f'{type(var_13.start_position).__module__}.{type(var_13.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_13.end_position).__module__}.{type(var_13.end_position).__qualname__}' == 'typesystem.base.Position'
    var_14 = var_11.__str__()
    var_15 = var_13.__repr__()
    assert var_15 == "Message(text='c$Yc^\\r%', code='custom', index=[1817854750117161950], position=Position(line_no=True, column_no=None, char_index=True))"
    var_16 = var_11.__repr__()
    assert var_16 == "ParseError(text=None, code='custom')"
    var_17 = var_11.items()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17) == 1
    var_18 = var_5.__repr__()
    assert var_18 == 'ValidationResult(value=None)'
    var_19 = module_0.ValidationError(text=var_17, code=var_17)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19) == 1
    var_20 = var_19.__eq__(var_19)
    assert var_20 is True
    module_0.ParseError(key=var_3, position=var_8)

def test_case_39():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = 'test_error'
    var_3 = module_0.ValidationError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = module_0.ValidationResult(error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = None
    var_6 = module_0.Message(text=var_2, index=var_5, start_position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'test_error'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = module_0.Position(var_6, var_5, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_7.column_no is None
    assert var_7.char_index is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_2, code=var_0, position=var_7, start_position=var_7)

def test_case_40():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = iter(var_1)
    var_3 = next(var_2)
    assert var_3 == 'test_value'
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = 'test_error'
    var_6 = module_0.ValidationError(text=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult(error=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)

def test_case_41():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'field_name'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = repr(var_5)
    var_7 = 'Error 1'
    var_8 = 'code1'
    var_9 = 'field1'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'Error 1'
    assert var_10.code == 'code1'
    assert var_10.index == ['field1']
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = 'Error 2'
    var_12 = 'code2'
    var_13 = 'field2'
    var_14 = module_0.Message(text=var_11, code=var_12, key=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'Error 2'
    assert var_14.code == 'code2'
    assert var_14.index == ['field2']
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = [var_10, var_14]
    var_16 = module_0.BaseError(messages=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 2
    var_17 = repr(var_16)
    var_18 = 2
    var_19 = 3
    var_20 = module_0.Position(var_17, var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no == "BaseError([Message(text='Error 1', code='code1', index=['field1']), Message(text='Error 2', code='code2', index=['field2'])])"
    assert var_20.column_no == 2
    assert var_20.char_index == 3
    var_21 = module_0.BaseError(text=var_0, code=var_1, position=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_21) == 1
    var_22 = repr(var_21)

def test_case_42():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = dict(var_3)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = module_0.Position(var_6, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 1
    assert var_9.column_no == 2
    assert var_9.char_index == 3
    var_10 = 'Error with position'
    var_11 = module_0.BaseError(text=var_10, position=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = 'First error'
    var_13 = 'first'
    var_14 = module_0.Message(text=var_12, code=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'First error'
    assert var_14.code == 'first'
    assert var_14.index == []
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = 'Second error'
    var_16 = 'second'
    var_17 = module_0.Message(text=var_15, code=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'Second error'
    assert var_17.code == 'second'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = [var_14, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_19) == 1
    var_20 = len(var_19)
    var_21 = dict(var_19)
    var_22 = 'Nested error'
    var_23 = 'nested'
    var_24 = 'a'
    var_25 = 'b'
    var_26 = [var_24, var_25]
    var_27 = module_0.Message(text=var_22, code=var_23, index=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Message'
    assert var_27.text == 'Nested error'
    assert var_27.code == 'nested'
    assert var_27.index == ['a', 'b']
    assert var_27.start_position is None
    assert var_27.end_position is None
    var_28 = [var_27]
    var_29 = module_0.BaseError(messages=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_29) == 1
    var_30 = dict(var_29)

def test_case_43():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'key1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error message'
    assert var_3.code == 'error_code'
    assert var_3.index == ['key1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error message'
    assert var_4.code == 'error_code'
    assert var_4.index == ['key1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Different message'
    var_6 = module_0.Message(text=var_5, code=var_1, key=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Different message'
    assert var_6.code == 'error_code'
    assert var_6.index == ['key1']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'different_code'
    var_8 = module_0.Message(text=var_0, code=var_7, key=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error message'
    assert var_8.code == 'different_code'
    assert var_8.index == ['key1']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_0, code=var_1, key=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'Error message'
    assert var_10.code == 'error_code'
    assert var_10.index == ['key2']
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = 1
    var_12 = 2
    var_13 = 3
    var_14 = module_0.Position(var_11, var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no == 1
    assert var_14.column_no == 2
    assert var_14.char_index == 3
    var_15 = module_0.Message(text=var_0, code=var_1, position=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'Error message'
    assert var_15.code == 'error_code'
    assert var_15.index == []
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.Position'
    var_16 = module_0.Message(text=var_0, code=var_1, position=var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'Error message'
    assert var_16.code == 'error_code'
    assert var_16.index == []
    assert f'{type(var_16.start_position).__module__}.{type(var_16.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_16.end_position).__module__}.{type(var_16.end_position).__qualname__}' == 'typesystem.base.Position'
    var_17 = 4
    var_18 = 5
    var_19 = 6
    var_20 = module_0.Position(var_17, var_18, var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no == 4
    assert var_20.column_no == 5
    assert var_20.char_index == 6
    var_21 = module_0.Message(text=var_0, code=var_1, position=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text == 'Error message'
    assert var_21.code == 'error_code'
    assert var_21.index == []
    assert f'{type(var_21.start_position).__module__}.{type(var_21.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_21.end_position).__module__}.{type(var_21.end_position).__qualname__}' == 'typesystem.base.Position'
    var_22 = module_0.Position(var_11, var_12, var_13)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert var_22.line_no == 1
    assert var_22.column_no == 2
    assert var_22.char_index == 3
    var_23 = module_0.Position(var_17, var_18, var_19)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Position'
    assert var_23.line_no == 4
    assert var_23.column_no == 5
    assert var_23.char_index == 6
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Message'
    assert var_24.text == 'Error message'
    assert var_24.code == 'error_code'
    assert var_24.index == []
    assert f'{type(var_24.start_position).__module__}.{type(var_24.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_24.end_position).__module__}.{type(var_24.end_position).__qualname__}' == 'typesystem.base.Position'
    var_25 = module_0.Message(text=var_0, code=var_1, start_position=var_22, end_position=var_23)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text == 'Error message'
    assert var_25.code == 'error_code'
    assert var_25.index == []
    assert f'{type(var_25.start_position).__module__}.{type(var_25.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_25.end_position).__module__}.{type(var_25.end_position).__qualname__}' == 'typesystem.base.Position'
    var_26 = 7
    var_27 = 8
    var_28 = 9
    var_29 = module_0.Position(var_26, var_27, var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Position'
    assert var_29.line_no == 7
    assert var_29.column_no == 8
    assert var_29.char_index == 9
    var_30 = module_0.Message(text=var_0, code=var_1, start_position=var_29, end_position=var_23)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.Message'
    assert var_30.text == 'Error message'
    assert var_30.code == 'error_code'
    assert var_30.index == []
    assert f'{type(var_30.start_position).__module__}.{type(var_30.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_30.end_position).__module__}.{type(var_30.end_position).__qualname__}' == 'typesystem.base.Position'
    var_31 = var_30.__repr__()
    assert var_31 == "Message(text='Error message', code='error_code', start_position=Position(line_no=7, column_no=8, char_index=9), end_position=Position(line_no=4, column_no=5, char_index=6))"
    var_32 = None
    var_33 = module_0.Position(var_32, var_32, var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.base.Position'
    assert var_33.line_no is None
    assert var_33.column_no is None
    assert var_33.char_index is None
    var_34 = module_0.Message(text=var_0, index=var_32, start_position=var_32)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.base.Message'
    assert var_34.text == 'Error message'
    assert var_34.code == 'custom'
    assert var_34.index == []
    assert var_34.start_position is None
    assert var_34.end_position is None