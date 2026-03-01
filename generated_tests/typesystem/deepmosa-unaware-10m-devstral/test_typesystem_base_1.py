# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 'L{!z*~!_}\n;;'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_1():
    var_0 = '1>Wc<?U=o{G'
    var_1 = module_0.ValidationError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

def test_case_2():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'HV!dfDI'
    module_0.ValidationError(code=var_0, messages=var_0)

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_5():
    var_0 = 1394
    var_1 = None
    var_2 = False
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1394
    assert var_3.column_no is None
    assert var_3.char_index is False
    var_4 = var_3.__repr__()
    assert var_4 == 'Position(line_no=1394, column_no=None, char_index=False)'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 1280923952461385056
    var_3 = None
    var_4 = module_0.Message(text=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    module_0.ValidationError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = var_2.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = module_0.ValidationError(text=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_1.__eq__(var_0)
    assert var_7 is False
    var_8 = var_5.__eq__(var_2)
    assert var_8 is False
    var_9 = var_5.__len__()
    assert var_9 == 1
    var_10 = var_5.__repr__()
    assert var_10 == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    module_0.ValidationError(code=var_9, messages=var_9)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = 'L{!z*~!_}\n;;'
    var_2 = module_0.BaseError(text=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_2.__getitem__(var_0)

def test_case_9():
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
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = module_0.ValidationError(text=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = module_0.ValidationResult(error=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert f'{type(var_8.error).__module__}.{type(var_8.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.error) == 1
    var_9 = iter(var_8)
    var_10 = next(var_9)
    assert var_10 is None
    var_11 = next(var_9)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is False
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = '1>Wc<?U=o{G'
    var_9 = module_0.ValidationError(text=var_5, code=var_8, position=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_9.__contains__(var_2)
    assert var_10 is False
    var_11 = var_6.__eq__(var_1)
    assert var_11 is True
    var_12 = var_9.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = var_9.__repr__()
    assert var_15 == 'ValidationError(text="Message(text=None, code=\'custom\')", code=\'1>Wc<?U=o{G\')'
    module_0.ValidationError(text=var_5, code=var_10, messages=var_10)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    module_0.ValidationError(messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
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
    module_0.ValidationError(key=var_0, messages=var_0)

def test_case_13():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is False
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_1.__hash__()
    assert var_5 == 1280923952461385056
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_3.__repr__()
    assert var_7 == 'Position(line_no=False, column_no=False, char_index=None)'
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = '1cZWb<?U=o{G'
    var_10 = module_0.ValidationError(text=var_6, code=var_9, position=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_1.__eq__(var_1)
    assert var_11 is True
    var_12 = var_10.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_10.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Position(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_6 = var_2.__hash__()
    assert var_6 == 1280923952461385056
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_0.__repr__()
    assert var_9 == 'None'
    var_10 = var_2.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_8.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = "z*l6F+g\nY'"
    var_13 = module_0.ValidationError(text=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.messages(add_prefix=var_7)
    var_15 = var_2.__eq__(var_9)
    assert var_15 is False
    module_0.ValidationError(code=var_0, position=var_0)

def test_case_15():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Position(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_6 = var_2.__hash__()
    assert var_6 == 1280923952461385056
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None

def test_case_16():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = var_1.__len__()
    assert var_2 == 28
    var_3 = '1cZWb<?U=o{G'
    var_4 = module_0.ValidationError(text=var_3, code=var_3, position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_2)
    assert var_5 is False
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    module_0.ValidationError(messages=var_3)

def test_case_18():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = '1>Wc<?U=o{G'
    var_4 = module_0.ValidationError(text=var_2, code=var_3, position=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_1.__eq__(var_1)
    assert var_5 is True
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationError(text="Message(text=None, code=\'custom\')", code=\'1>Wc<?U=o{G\')'

def test_case_19():
    var_0 = '/W#\n'
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '/W#\n'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = None
    var_5 = var_2.__eq__(var_4)
    assert var_5 is False
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text='/W#\\n', code='custom')"
    var_7 = 'DDD7}ui\x0bgA{B^`]R'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_7, code=var_7, key=var_7, index=var_0, position=var_4)

def test_case_20():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_21():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = iter(var_1)
    var_3 = 'test_error'
    var_4 = module_0.ValidationError(text=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = module_0.ValidationResult(error=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = iter(var_5)
    var_7 = next(var_6)
    assert var_7 is None
    var_8 = next(var_6)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Position(var_0, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is False
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_2.__hash__()
    assert var_6 == 1280923952461385056
    var_7 = 'T]m^Gp'
    var_8 = module_0.Message(text=var_7, code=var_0, key=var_0, start_position=var_4, end_position=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'T]m^Gp'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = var_4.__eq__(var_8)
    assert var_9 is False
    var_10 = var_8.__repr__()
    assert var_10 == "Message(text='T]m^Gp', code='custom', position=Position(line_no=None, column_no=None, char_index=False))"
    var_11 = var_8.__repr__()
    assert var_11 == "Message(text='T]m^Gp', code='custom', position=Position(line_no=None, column_no=None, char_index=False))"
    var_12 = var_4.__repr__()
    assert var_12 == 'Position(line_no=None, column_no=None, char_index=False)'
    var_13 = []
    module_0.ValidationError(key=var_7, position=var_0, messages=var_13)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is False
    assert var_3.char_index is None
    var_4 = module_0.Message(text=var_0, key=var_2, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == [False]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='custom', index=[False])"
    var_6 = '~tfO7te6'
    var_7 = module_0.ValidationError(text=var_5, code=var_6, position=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.__contains__(var_2)
    assert var_8 is False
    var_9 = var_4.__eq__(var_1)
    assert var_9 is False
    var_10 = var_7.__eq__(var_0)
    assert var_10 is False
    var_11 = var_7.__repr__()
    assert var_11 == 'ValidationError(text="Message(text=None, code=\'custom\', index=[False])", code=\'~tfO7te6\')'
    var_12 = var_8.__eq__(var_8)
    assert var_12 is True
    module_0.ValidationError(code=var_0, key=var_0, messages=var_8)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = b'\x8e\x93\x07>\xda^|~'
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == b'\x8e\x93\x07>\xda^|~'
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__hash__()
    assert var_4 == 1280923952461385056
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text=None, code='custom', start_position=b'\\x8e\\x93\\x07>\\xda^|~', end_position=None)"
    var_6 = module_0.Message(text=var_4, end_position=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 1280923952461385056
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position == 1280923952461385056
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_7.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    module_0.ValidationError(text=var_0, messages=var_0)

def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = '1>Wc<?U=o{G'
    var_5 = module_0.ValidationError(text=var_2, code=var_4, position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_3.__eq__(var_1)
    assert var_6 is True
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_5.__repr__()
    assert var_9 == 'ValidationError(text="Message(text=None, code=\'custom\')", code=\'1>Wc<?U=o{G\')'

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 'REzX.cOs~"/F\'u\x0bi='
    var_1 = None
    var_2 = module_0.Message(text=var_1, key=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = ''
    var_4 = module_0.ValidationError(text=var_0, code=var_3, position=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_4.__str__()
    assert var_6 == 'REzX.cOs~"/F\'u\x0bi='
    module_0.ValidationError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = ''
    var_7 = module_0.ValidationError(text=var_4, code=var_6, position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.__iter__()
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    module_0.ValidationError(messages=var_6)

def test_case_28():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__hash__()
    assert var_4 == 1280923952461385056
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.Message(text=var_1, key=var_1, start_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_7.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = '1cZWb<?U=o{G'
    var_11 = module_0.ValidationError(text=var_5, code=var_10, position=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_6.__eq__(var_2)
    assert var_12 is True
    var_13 = var_11.__eq__(var_1)
    assert var_13 is False
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_3, error=var_11)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = b'\x8e\x93\x07>\xda^|~'
    var_3 = module_0.Message(text=var_0, index=var_0, position=var_0, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == b'\x8e\x93\x07>\xda^|~'
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_1.__hash__()
    assert var_5 == 1280923952461385056
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.Message(text=var_5, end_position=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 1280923952461385056
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position == 1280923952461385056
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_8.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    module_0.ValidationError(text=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Position(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_6 = var_2.__hash__()
    assert var_6 == 1280923952461385056
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.Message(text=var_1, key=var_1, start_position=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = module_0.Message(text=var_7, code=var_1, key=var_4, index=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == "Message(text=None, code='custom')"
    assert var_9.code == 'custom'
    assert f'{type(var_9.index).__module__}.{type(var_9.index).__qualname__}' == 'builtins.list'
    assert len(var_9.index) == 1
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = True
    var_11 = module_0.ValidationResult(error=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is True
    var_12 = var_2.__repr__()
    assert var_12 == "Message(text=None, code='custom')"
    var_13 = var_11.__repr__()
    assert var_13 == 'ValidationResult(error=True)'
    module_0.ValidationError(text=var_0, key=var_7)

def test_case_31():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Position(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_6 = var_2.__hash__()
    assert var_6 == 1280923952461385056
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, code=var_1, position=var_4, start_position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__hash__()
    assert var_3 == 1280923952461385056
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.Message(text=var_1, key=var_1, start_position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'typesystem.base.Message'
    assert var_5.code == 'custom'
    assert f'{type(var_5.index).__module__}.{type(var_5.index).__qualname__}' == 'builtins.list'
    assert len(var_5.index) == 1
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Message'
    assert var_5.end_position is None
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_6.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = '1cZb<?=o{G'
    var_10 = module_0.ValidationError(text=var_4, code=var_9, position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_10.messages()
    var_12 = var_5.__eq__(var_1)
    assert var_12 is False
    var_13 = var_10.__eq__(var_0)
    assert var_13 is False
    var_14 = var_10.__len__()
    assert var_14 == 1
    module_0.ValidationError(messages=var_9)

def test_case_33():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__hash__()
    assert var_4 == 1280923952461385056
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_0.__repr__()
    assert var_7 == 'None'
    var_8 = var_6.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = '1cZWb<?U=o{G'
    var_10 = module_0.ValidationError(text=var_5, code=var_9, position=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_10.messages()
    var_12 = var_2.__eq__(var_2)
    assert var_12 is True
    var_13 = var_10.__eq__(var_10)
    assert var_13 is True
    var_14 = module_0.ValidationError(text=var_8, code=var_4, key=var_7)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is False
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_4.__repr__()
    assert var_7 == 'False'
    var_8 = var_1.__hash__()
    assert var_8 == 1280923952461385056
    var_9 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is None
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_9.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = '1>Wc<?U=o{G'
    var_12 = module_0.ValidationError(text=var_6, code=var_11, position=var_1)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_12.__contains__(var_5)
    assert var_13 is False
    var_14 = var_9.__eq__(var_1)
    assert var_14 is True
    var_15 = var_12.__eq__(var_0)
    assert var_15 is False
    var_16 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_12.__repr__()
    assert var_17 == 'ValidationError(text="Message(text=None, code=\'custom\')", code=\'1>Wc<?U=o{G\')'
    module_0.ValidationError(position=var_13, messages=var_13)

def test_case_35():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is False
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_1.__hash__()
    assert var_7 == 1280923952461385056
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_5, index=var_0, position=var_3, end_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.Message(text=var_0, index=var_0, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__repr__()
    assert var_7 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_8 = module_0.ValidationError(text=var_7, key=var_5, position=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == '{"Message(text=None, code=\'custom\')": \'Position(line_no=None, column_no=None, char_index=None)\'}'
    var_10 = var_6.__hash__()
    assert var_10 == 1280923952461385056
    var_11 = var_6.__eq__(var_0)
    assert var_11 is False
    var_12 = var_8.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult(value=var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value == '{"Message(text=None, code=\'custom\')": \'Position(line_no=None, column_no=None, char_index=None)\'}'
    assert var_13.error is None
    var_14 = var_8.__repr__()
    assert var_14 == 'ValidationError([Message(text=\'Position(line_no=None, column_no=None, char_index=None)\', code=\'custom\', index=["Message(text=None, code=\'custom\')"], position=Position(line_no=None, column_no=None, char_index=None))])'
    module_0.ValidationError(code=var_0, key=var_0)

def test_case_37():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_0, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is False
    assert var_3.char_index is False
    var_4 = -1582
    var_5 = module_0.Position(var_0, var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is None
    assert var_5.column_no == -1582
    assert var_5.char_index is None
    var_6 = var_3.__eq__(var_5)
    assert var_6 is False
    var_7 = var_1.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.Message(text=var_7, index=var_0, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == "Message(text=None, code='custom')"
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_1.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = module_0.ValidationError(text=var_10, position=var_0, messages=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_11.__repr__()
    assert var_12 == 'ValidationError(text="Message(text=None, code=\'custom\')", code=\'custom\')'
    var_13 = var_11.__contains__(var_0)
    assert var_13 is False
    var_14 = var_13.__eq__(var_13)
    assert var_14 is True
    var_15 = var_13.__eq__(var_6)
    assert var_15 is True
    var_16 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = True
    var_4 = module_0.Position(var_3, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is True
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = 'Ut\n@$)`Y Fy<$R9!6'
    var_8 = module_0.Message(text=var_7, code=var_0, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Ut\n@$)`Y Fy<$R9!6'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_4.__repr__()
    assert var_9 == 'Position(line_no=True, column_no=None, char_index=None)'
    var_10 = [var_8, var_8]
    var_11 = module_0.ValidationError(key=var_0, position=var_0, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = var_11.__str__()
    assert var_12 == "{'': 'Ut\\n@$)`Y Fy<$R9!6'}"
    var_13 = var_1.__hash__()
    assert var_13 == 1280923952461385056
    var_14 = var_8.__eq__(var_0)
    assert var_14 is False
    var_15 = var_11.__len__()
    assert var_15 == 1
    var_16 = var_15.__eq__(var_0)
    var_17 = module_0.ValidationResult(value=var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value == 1
    assert var_17.error is None
    var_18 = var_11.__repr__()
    assert var_18 == "ValidationError([Message(text='Ut\\n@$)`Y Fy<$R9!6', code='custom'), Message(text='Ut\\n@$)`Y Fy<$R9!6', code='custom')])"
    module_0.ValidationError(text=var_0)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = 291
    var_5 = module_0.Position(var_4, var_2, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 291
    assert var_5.column_no is False
    assert var_5.char_index is False
    var_6 = var_3.__eq__(var_5)
    assert var_6 is False
    var_7 = var_5.__eq__(var_2)
    assert var_7 is False
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.Message(text=var_7, end_position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is False
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_9.__repr__()
    assert var_10 == "Message(text=False, code='custom')"
    module_0.ValidationError(key=var_8, position=var_0)

def test_case_40():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3._messages
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 'Error 1'
    var_7 = 'code1'
    var_8 = 'key1'
    var_9 = module_0.Message(text=var_6, code=var_7, key=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 1'
    assert var_9.code == 'code1'
    assert var_9.index == ['key1']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = 'Error 2'
    var_11 = 'code2'
    var_12 = 'key2'
    var_13 = module_0.Message(text=var_10, code=var_11, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'Error 2'
    assert var_13.code == 'code2'
    assert var_13.index == ['key2']
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = [var_9, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 2
    var_16 = var_15._messages
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = dict(var_15)
    var_19 = 'users'
    var_20 = 0
    var_21 = 'username'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.Message(text=var_6, code=var_7, index=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'Error 1'
    assert var_23.code == 'code1'
    assert var_23.index == ['users', 0, 'username']
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = 1
    var_25 = 'email'
    var_26 = [var_19, var_24, var_25]
    var_27 = module_0.Message(text=var_10, code=var_11, index=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Message'
    assert var_27.text == 'Error 2'
    assert var_27.code == 'code2'
    assert var_27.index == ['users', 1, 'email']
    assert var_27.start_position is None
    assert var_27.end_position is None
    var_28 = [var_23, var_27]
    var_29 = module_0.BaseError(messages=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_29) == 1
    var_30 = var_29._messages
    var_31 = len(var_30)
    assert var_31 == 2
    var_32 = dict(var_29)
    var_33 = 2
    var_34 = 3
    var_35 = module_0.Position(var_24, var_33, var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.base.Position'
    assert var_35.line_no == 1
    assert var_35.column_no == 2
    assert var_35.char_index == 3
    var_36 = 'Error with position'
    var_37 = module_0.BaseError(text=var_36, position=var_35)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_37) == 1
    var_38 = module_0.Position(var_24, var_33, var_34)
    assert f'{type(var_38).__module__}.{type(var_38).__qualname__}' == 'typesystem.base.Position'
    assert var_38.line_no == 1
    assert var_38.column_no == 2
    assert var_38.char_index == 3
    var_39 = 5
    var_40 = 8
    var_41 = module_0.Position(var_24, var_39, var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.base.Position'
    assert var_41.line_no == 1
    assert var_41.column_no == 5
    assert var_41.char_index == 8
    var_42 = 'Error with positions'
    var_43 = module_0.BaseError(text=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_43) == 1
    var_44 = 'Error without code'
    var_45 = module_0.BaseError(text=var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_45) == 1
    var_46 = 'Error'
    var_47 = 'code'
    var_48 = module_0.BaseError(text=var_46, code=var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_48) == 1
    var_49 = module_0.BaseError(text=var_46, code=var_47)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_49) == 1
    var_50 = module_0.BaseError(text=var_46, code=var_47)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_50) == 1
    with pytest.raises(TypeError):
        var_51 = hash(var_30)