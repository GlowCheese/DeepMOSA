# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = '#\nq+i,VZrXtPP$\nZ'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '#\nq+i,VZrXtPP$\nZ'
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = module_0.Position(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is False
    assert var_4.char_index is False
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == 'Position(line_no=False, column_no=False, char_index=False)'
    var_6.__getitem__(var_0)

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, index=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_2():
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_3():
    var_0 = '\x0b{.\r<WC}I:\\mIm3Gzr'
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0, key=var_0, messages=var_0)

def test_case_4():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = var_1.__iter__()

def test_case_5():
    var_0 = '#"\nq+i,wrq3"tPP$\n$'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == '#"\nq+i,wrq3"tPP$\n$'
    assert var_1.error is None

def test_case_6():
    var_0 = None
    var_1 = 2566
    var_2 = 4029
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no == 2566
    assert var_3.char_index == 4029
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_0)

def test_case_7():
    var_0 = '#"\nq+i,VrX3"tPP$\n$'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == '#"\nq+i,VrX3"tPP$\n$'
    assert var_1.error is None
    var_2 = module_0.ValidationResult()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__bool__()
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '#"\nq+i,VrX3"tPP$\n$'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = var_3.__str__()
    assert var_4 == 'KeysView(ValidationError(text=\'#"\\nq+i,VrX3"tPP$\\n$\', code=\'custom\'))'
    var_5 = module_0.ParseError(messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_3.__repr__()
    assert var_6 == 'KeysView(ValidationError(text=\'#"\\nq+i,VrX3"tPP$\\n$\', code=\'custom\'))'
    var_7 = var_2.__str__()
    assert var_7 == '[Message(text=\'#"\\nq+i,VrX3"tPP$\\n$\', code=\'custom\')]'
    module_0.ValidationError()

def test_case_9():
    var_0 = 'jqT#Aov;\n"D'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '#"\nq+i,VrXtPP$\n$'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()
    var_3 = var_1.__str__()
    assert var_3 == '#"\nq+i,VrXtPP$\n$'
    var_3.values()

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = '#"\nq+i,VrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#"\nq+i,VrXtPP$\n$'
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=\'#"\\nq+i,VrXtPP$\\n$\')'
    var_5 = var_2.__iter__()
    var_6 = var_2.__str__()
    assert var_6 == '#"\nq+i,VrXtPP$\n$'
    var_7 = var_2.get(var_1)
    var_8 = var_7.__eq__(var_1)
    assert var_8 is True
    var_9 = var_2.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = var_9.__eq__(var_1)
    var_6.values()

def test_case_12():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = []
    var_3 = None
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_2, start_position=var_3, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5, start_position=var_3, end_position=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error'
    assert var_6.code == 'custom'
    assert f'{type(var_6.index).__module__}.{type(var_6.index).__qualname__}' == 'builtins.list'
    assert len(var_6.index) == 1
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_4 == var_6

def test_case_13():
    var_0 = 'fkQXtc"GFC^|T][+oj'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, index=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'fkQXtc"GFC^|T][+oj'
    assert var_2.code == 'custom'
    assert var_2.index == ['fkQXtc"GFC^|T][+oj']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_3, messages=var_3)

def test_case_14():
    var_0 = '#"\nq+i,VrX3"tPP$\n$'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_15():
    var_0 = '(=E3c@5 X>lQY:\x0bP['
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    assert var_2 == -1077623326240478037

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'SoA\x0b~i'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "ValidationError(text='SoA\\x0b~i', code='custom')"
    var_3 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error == 'SoA\x0b~i'
    var_0.values()

def test_case_17():
    var_0 = 'od'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_4 = var_3.__eq__(var_2)
    var_5 = None
    var_6 = var_1.__eq__(var_5)
    assert var_6 is False
    var_7 = var_1.__eq__(var_5)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = None
    var_10 = var_1.messages()
    var_11 = var_1.__iter__()
    var_12 = var_7.__repr__()
    assert var_12 == 'False'
    var_13 = var_1.get(var_9)
    var_14 = var_1.__len__()
    assert var_14 == 1
    var_15 = var_14.__repr__()
    assert var_15 == '1'
    var_16 = module_0.ValidationResult()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '#"\nq+i,VrX3"tPP$\n$'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = module_0.ParseError(messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_5 = var_2.__str__()
    assert var_5 == '[Message(text=\'#"\\nq+i,VrX3"tPP$\\n$\', code=\'custom\')]'
    module_0.ValidationError()

def test_case_19():
    var_0 = '#\n+i,VrXPP\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_0.__repr__()
    assert var_3 == "'#\\n+i,VrXPP\\n$'"
    var_4 = var_2.__len__()
    assert var_4 == 1
    var_5 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_2.messages()
    var_8 = var_5.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_2.__eq__(var_7)
    assert var_9 is False
    var_10 = var_2.__repr__()
    assert var_10 == "ValidationError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_11 = var_5.__eq__(var_5)
    assert var_11 is True
    var_12 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is None
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = var_12.__eq__(var_1)
    assert var_13 is False
    var_14 = var_12.__repr__()
    assert var_14 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_15 = var_2.__repr__()
    assert var_15 == "ValidationError(text='#\\n+i,VrXPP\\n$', code='custom')"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_1, key=var_0, position=var_3, messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    module_0.ValidationError(key=var_0, position=var_1, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = '#"\nq+i,VrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#"\nq+i,VrXtPP$\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#"\nq+i,VrXtPP$\n$'
    var_5 = var_2.__eq__(var_1)
    assert var_5 is False
    var_6 = module_0.Position(var_5, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is False
    assert var_6.column_no is None
    assert var_6.char_index is None
    var_4.values()

def test_case_22():
    var_0 = '#"\nq+i,VrXtPP$\n$'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == '#"\nq+i,VrXtPP$\n$'
    assert var_2.error is None
    var_3 = var_1.__len__()
    assert var_3 == 1
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_2, error=var_2)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\nq+i,VZrXtPP$\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\nq+i,VZrXtPP$\n$'
    var_5 = var_3.__repr__()
    assert var_5 == "ValidationResult(value='#\\nq+i,VZrXtPP$\\n$')"
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == '#\nq+i,VZrXtPP$\n$'
    assert var_6.end_position is None
    var_7 = True
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom', start_position='#\\nq+i,VZrXtPP$\\n$', end_position=None)"
    var_9 = var_2.messages()
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_6.__eq__(var_1)
    assert var_11 is False
    var_12 = module_0.Position(var_7, var_1, var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is True
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = var_12.__eq__(var_1)
    assert var_13 is False
    var_14 = var_2.__repr__()
    assert var_14 == "ValidationError(text='#\\nq+i,VZrXtPP$\\n$', code='custom')"
    var_15 = module_0.BaseError(key=var_1, messages=var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = '7HUbqg!qe4'
    var_17 = module_0.ValidationError(text=var_16, code=var_1, key=var_4, messages=var_1)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17) == 1
    var_18 = var_17.__str__()
    assert var_18 == "{'#\\nq+i,VZrXtPP$\\n$': '7HUbqg!qe4'}"
    var_19 = var_15.__contains__(var_1)
    assert var_19 is False
    var_20 = var_19.__eq__(var_17)
    var_21 = var_2.keys()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_21) == 1
    var_12.keys()

def test_case_24():
    var_0 = None
    var_1 = -3128
    var_2 = 2019
    var_3 = module_0.Position(var_1, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == -3128
    assert var_3.column_no == 2019
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=-3128, column_no=2019, char_index=None)'

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = False
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.Position(var_2, var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_5.__repr__()
    assert var_7 == 'Position(line_no=False, column_no=None, char_index=None)'
    module_0.ValidationError(code=var_3)

def test_case_26():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == '#\nq+i,VZrXtPP$\n$'
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text=None, code='custom', start_position='#\\nq+i,VZrXtPP$\\n$', end_position=None)"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = '#\n+i,VrXPP\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\n+i,VrXPP\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\n+i,VrXPP\n$'
    var_5 = var_0.__repr__()
    assert var_5 == "'#\\n+i,VrXPP\\n$'"
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == '#\n+i,VrXPP\n$'
    assert var_6.end_position is None
    var_7 = False
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom', start_position='#\\n+i,VrXPP\\n$', end_position=None)"
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = var_6.__eq__(var_1)
    assert var_10 is False
    var_11 = module_0.Position(var_7, var_1, var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is False
    assert var_11.column_no is None
    assert var_11.char_index is None
    var_12 = var_11.__eq__(var_1)
    assert var_12 is False
    var_13 = var_11.__repr__()
    assert var_13 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_14 = var_2.__repr__()
    assert var_14 == "ValidationError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_15 = var_2.__iter__()
    var_16 = var_6.__hash__()
    assert var_16 == 9199009871729566954
    var_17 = module_0.BaseError(text=var_4, code=var_6, key=var_6, position=var_15, messages=var_1)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = None
    var_19 = None
    var_20 = module_0.ValidationError(text=var_15, position=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20) == 1
    var_21 = var_17.__contains__(var_19)
    assert var_21 is False
    var_22 = var_17.__eq__(var_2)
    assert var_22 is False
    var_22.get(var_18)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\nq+i,VZrXtPP$\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\nq+i,VZrXtPP$\n$'
    var_5 = var_0.__repr__()
    assert var_5 == "'#\\nq+i,VZrXtPP$\\n$'"
    var_6 = var_3.__repr__()
    assert var_6 == "ValidationResult(value='#\\nq+i,VZrXtPP$\\n$')"
    var_7 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position == '#\nq+i,VZrXtPP$\n$'
    assert var_7.end_position is None
    var_8 = var_2.messages()
    var_9 = var_2.messages(add_prefix=var_5)
    var_10 = var_2.__eq__(var_9)
    assert var_10 is False
    var_11 = var_7.__eq__(var_1)
    assert var_11 is False
    var_12 = -3128
    var_13 = 2019
    var_14 = module_0.Position(var_12, var_13, var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no == -3128
    assert var_14.column_no == 2019
    assert var_14.char_index is None
    var_15 = var_14.__eq__(var_14)
    assert var_15 is True
    var_16 = var_14.__repr__()
    assert var_16 == 'Position(line_no=-3128, column_no=2019, char_index=None)'
    var_17 = var_2.__repr__()
    assert var_17 == "ValidationError(text='#\\nq+i,VZrXtPP$\\n$', code='custom')"
    var_18 = var_7.__repr__()
    assert var_18 == "Message(text=None, code='custom', start_position='#\\nq+i,VZrXtPP$\\n$', end_position=None)"
    module_0.ValidationError(text=var_1)

def test_case_29():
    var_0 = '#\n+i,VZrX"tPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\n+i,VZrX"tPP$\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\n+i,VZrX"tPP$\n$'
    var_5 = var_3.__repr__()
    assert var_5 == 'ValidationResult(value=\'#\\n+i,VZrX"tPP$\\n$\')'
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = False
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_2.messages()
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_3.__iter__()
    var_12 = var_6.__eq__(var_1)
    assert var_12 is False
    var_13 = module_0.Position(var_7, var_1, var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is False
    assert var_13.column_no is None
    assert var_13.char_index is None
    var_14 = var_13.__eq__(var_1)
    assert var_14 is False
    var_15 = var_13.__repr__()
    assert var_15 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_16 = var_2.__repr__()
    assert var_16 == 'ValidationError(text=\'#\\n+i,VZrX"tPP$\\n$\', code=\'custom\')'
    var_17 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_1, messages=var_17)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = '#\n+i,VrXPP\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\n+i,VrXPP\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\n+i,VrXPP\n$'
    var_5 = var_0.__repr__()
    assert var_5 == "'#\\n+i,VrXPP\\n$'"
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == '#\n+i,VrXPP\n$'
    assert var_6.end_position is None
    var_7 = False
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom', start_position='#\\n+i,VrXPP\\n$', end_position=None)"
    var_9 = var_2.messages()
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_6.__eq__(var_1)
    assert var_11 is False
    var_12 = module_0.Position(var_7, var_1, var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is False
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = var_12.__eq__(var_1)
    assert var_13 is False
    var_14 = var_12.__repr__()
    assert var_14 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_15 = var_2.__repr__()
    assert var_15 == "ValidationError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_16 = module_0.BaseError(key=var_1, messages=var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = var_2.__iter__()
    var_18 = var_16.__repr__()
    assert var_18 == "BaseError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_19 = var_6.__hash__()
    assert var_19 == 9199009871729566954
    var_20 = module_0.BaseError(text=var_4, code=var_6, key=var_6, position=var_17, messages=var_1)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = var_20.__repr__()
    var_22 = var_17.__hash__()
    var_23 = module_0.BaseError(position=var_1, messages=var_9)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_23) == 1
    var_24 = None
    var_25 = None
    module_0.ValidationError(text=var_1, code=var_1, key=var_24, position=var_25, messages=var_25)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\nq+i,VZrXtPP$\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\nq+i,VZrXtPP$\n$'
    var_5 = var_2.__contains__(var_1)
    assert var_5 is False
    var_6 = var_5.__repr__()
    assert var_6 == 'False'
    var_7 = var_3.__repr__()
    assert var_7 == "ValidationResult(value='#\\nq+i,VZrXtPP$\\n$')"
    var_8 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == '#\nq+i,VZrXtPP$\n$'
    assert var_8.end_position is None
    var_9 = False
    var_10 = var_8.__repr__()
    assert var_10 == "Message(text=None, code='custom', start_position='#\\nq+i,VZrXtPP$\\n$', end_position=None)"
    var_11 = var_2.messages()
    var_12 = var_2.__eq__(var_1)
    assert var_12 is False
    var_13 = var_8.__eq__(var_1)
    assert var_13 is False
    var_14 = module_0.Position(var_9, var_1, var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no is False
    assert var_14.column_no is None
    assert var_14.char_index is None
    var_15 = var_14.__eq__(var_1)
    assert var_15 is False
    var_16 = var_14.__repr__()
    assert var_16 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_17 = module_0.BaseError(key=var_1, messages=var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = var_2.__iter__()
    var_19 = var_18.__eq__(var_1)
    var_20 = var_14.__repr__()
    assert var_20 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_21 = var_17.__repr__()
    assert var_21 == "BaseError(text='#\\nq+i,VZrXtPP$\\n$', code='custom')"
    var_22 = var_8.__hash__()
    assert var_22 == 9199009871729566954
    var_23 = module_0.BaseError(text=var_4, code=var_5, key=var_5, position=var_18, messages=var_1)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_23) == 1
    module_0.ValidationError(text=var_20, messages=var_9)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = ')t4ACf1; hQ?'
    var_1 = module_0.BaseError(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()
    var_2.values()

def test_case_33():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = 2426
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index == 2426
    var_4 = None
    var_5 = module_0.ValidationError(text=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value == '#\nq+i,VZrXtPP$\n$'
    assert var_6.error is None
    var_7 = var_5.__str__()
    assert var_7 == '#\nq+i,VZrXtPP$\n$'
    var_8 = var_6.__repr__()
    assert var_8 == "ValidationResult(value='#\\nq+i,VZrXtPP$\\n$')"
    var_9 = var_5.__str__()
    assert var_9 == '#\nq+i,VZrXtPP$\n$'
    var_10 = var_5.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == "KeysView(ValidationError(text='#\\nq+i,VZrXtPP$\\n$', code='custom'))"
    var_12 = '}XOQ`C.:R&eCgwgKN{p'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_12, code=var_4, position=var_10, start_position=var_10, end_position=var_4)

def test_case_34():
    var_0 = "}_~sM:=8w'2{3TN"
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "}_~sM:=8w'2{3TN"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    var_4 = None
    var_5 = [var_2, var_2, var_2]
    var_6 = module_0.ValidationError(text=var_4, key=var_3, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_6.__str__()
    assert var_8 == '{\'\': "}_~sM:=8w\'2{3TN"}'
    var_9 = var_6.__repr__()
    assert var_9 == 'ValidationError([Message(text="}_~sM:=8w\'2{3TN", code=\'custom\'), Message(text="}_~sM:=8w\'2{3TN", code=\'custom\'), Message(text="}_~sM:=8w\'2{3TN", code=\'custom\')])'
    var_10 = module_0.Message(text=var_9, code=var_0, start_position=var_1, end_position=var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'ValidationError([Message(text="}_~sM:=8w\'2{3TN", code=\'custom\'), Message(text="}_~sM:=8w\'2{3TN", code=\'custom\'), Message(text="}_~sM:=8w\'2{3TN", code=\'custom\')])'
    assert var_10.code == "}_~sM:=8w'2{3TN"
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_10.__repr__()
    assert var_11 == 'Message(text=\'ValidationError([Message(text="}_~sM:=8w\\\'2{3TN", code=\\\'custom\\\'), Message(text="}_~sM:=8w\\\'2{3TN", code=\\\'custom\\\'), Message(text="}_~sM:=8w\\\'2{3TN", code=\\\'custom\\\')])\', code="}_~sM:=8w\'2{3TN")'
    var_12 = var_10.__repr__()
    assert var_12 == 'Message(text=\'ValidationError([Message(text="}_~sM:=8w\\\'2{3TN", code=\\\'custom\\\'), Message(text="}_~sM:=8w\\\'2{3TN", code=\\\'custom\\\'), Message(text="}_~sM:=8w\\\'2{3TN", code=\\\'custom\\\')])\', code="}_~sM:=8w\'2{3TN")'
    var_13 = var_6.messages()
    var_14 = var_6.__eq__(var_3)
    assert var_14 is False
    var_15 = module_0.Position(var_3, var_3, var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no is None
    assert var_15.column_no is None
    assert var_15.char_index is None
    var_16 = var_7.__eq__(var_3)
    var_17 = var_15.__repr__()
    assert var_17 == 'Position(line_no=None, column_no=None, char_index=None)'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_1, key=var_9, position=var_11)

def test_case_35():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = var_2.__str__()
    assert var_4 == '#\nq+i,VZrXtPP$\n$'
    var_5 = var_3.__repr__()
    assert var_5 == "ValidationResult(error=ValidationError(text='#\\nq+i,VZrXtPP$\\n$', code='custom'))"
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == '#\nq+i,VZrXtPP$\n$'
    assert var_6.end_position is None
    var_7 = var_4.__iter__()
    var_8 = False
    var_9 = var_6.__repr__()
    assert var_9 == "Message(text=None, code='custom', start_position='#\\nq+i,VZrXtPP$\\n$', end_position=None)"
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_6.__eq__(var_1)
    assert var_11 is False
    var_12 = module_0.Position(var_8, var_1, var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is False
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = var_12.__eq__(var_1)
    assert var_13 is False
    var_14 = var_12.__repr__()
    assert var_14 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_15 = '7XHUbqga!qe4'
    var_16 = module_0.ValidationError(text=var_15, code=var_1, key=var_4, messages=var_1)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16) == 1

def test_case_36():
    var_0 = '#\nq+i,VZrXtPP$\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\nq+i,VZrXtPP$\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\nq+i,VZrXtPP$\n$'
    var_5 = var_2.__contains__(var_1)
    assert var_5 is False
    var_6 = var_5.__repr__()
    assert var_6 == 'False'
    var_7 = var_3.__repr__()
    assert var_7 == "ValidationResult(value='#\\nq+i,VZrXtPP$\\n$')"
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_7, index=var_5)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = '#\n+i,VrXPP\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\n+i,VrXPP\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\n+i,VrXPP\n$'
    var_5 = var_0.__repr__()
    assert var_5 == "'#\\n+i,VrXPP\\n$'"
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == '#\n+i,VrXPP\n$'
    assert var_6.end_position is None
    var_7 = False
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom', start_position='#\\n+i,VrXPP\\n$', end_position=None)"
    var_9 = var_2.messages()
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_6.__eq__(var_1)
    assert var_11 is False
    var_12 = module_0.Position(var_7, var_1, var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is False
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = var_12.__eq__(var_1)
    assert var_13 is False
    var_14 = var_12.__repr__()
    assert var_14 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_15 = var_2.__repr__()
    assert var_15 == "ValidationError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_16 = module_0.BaseError(key=var_1, messages=var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = var_16.messages()
    var_18 = var_16.__repr__()
    assert var_18 == "BaseError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_19 = var_6.__hash__()
    assert var_19 == 9199009871729566954
    var_20 = None
    var_21 = None
    var_22 = var_16.messages(add_prefix=var_17)
    var_23 = module_0.ValidationError(text=var_17, position=var_21)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_23) == 1
    var_24 = var_16.values()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_24) == 1
    var_25 = var_16.__contains__(var_21)
    assert var_25 is False
    var_26 = var_23.__eq__(var_2)
    assert var_26 is False
    var_26.get(var_20)

def test_case_38():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 2
    var_4 = 10
    var_5 = module_0.Position(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 2
    assert var_5.column_no == 2
    assert var_5.char_index == 10
    var_6 = 'Error'
    var_7 = 'custom'
    var_8 = []
    var_9 = module_0.Message(text=var_6, code=var_7, index=var_8, start_position=var_2, end_position=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error'
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.end_position).__module__}.{type(var_9.end_position).__qualname__}' == 'typesystem.base.Position'
    var_10 = []
    var_11 = module_0.Message(text=var_6, code=var_7, index=var_10, start_position=var_5, end_position=var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'Error'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.Position'
    var_12 = var_9 == var_11
    assert var_12 is False

def test_case_39():
    var_0 = '#\n+i,VrXPP\n$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == '#\n+i,VrXPP\n$'
    assert var_3.error is None
    var_4 = var_2.__str__()
    assert var_4 == '#\n+i,VrXPP\n$'
    var_5 = var_0.__repr__()
    assert var_5 == "'#\\n+i,VrXPP\\n$'"
    var_6 = module_0.Message(text=var_1, code=var_1, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == '#\n+i,VrXPP\n$'
    assert var_6.end_position is None
    var_7 = False
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom', start_position='#\\n+i,VrXPP\\n$', end_position=None)"
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = var_6.__eq__(var_1)
    assert var_10 is False
    var_11 = module_0.Position(var_7, var_1, var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is False
    assert var_11.column_no is None
    assert var_11.char_index is None
    var_12 = var_11.__eq__(var_1)
    assert var_12 is False
    var_13 = var_11.__repr__()
    assert var_13 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_14 = var_2.__repr__()
    assert var_14 == "ValidationError(text='#\\n+i,VrXPP\\n$', code='custom')"
    var_15 = var_2.__iter__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_15, index=var_1, position=var_15, end_position=var_9)

def test_case_40():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = 5
    var_5 = module_0.Position(var_0, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 5
    assert var_5.char_index == 3
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_41():
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
    var_7 = 0
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_4, code=var_5, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'code2'
    assert var_9.index == ['field2', 0]
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_3, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = len(var_8)
    assert var_12 == 2
    var_13 = var_8[0]
    var_14 = bool(var_8[0] == var_3)
    with pytest.raises(TypeError):
        var_15 = var_14[1]

def test_case_42():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = iter(var_0)
    var_2 = next(var_1)
    assert var_2 is None
    var_3 = next(var_1)
    assert var_3 is None