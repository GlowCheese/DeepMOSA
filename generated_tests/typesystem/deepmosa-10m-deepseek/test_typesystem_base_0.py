# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0


def test_case_0():
    var_0 = None
    var_1 = -2004
    var_2 = False
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no == -2004
    assert var_3.char_index is False
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False

def test_case_1():
    var_0 = ''
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ''
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == ''
    assert var_1.end_position == ''
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text='', code='custom', position='')"

def test_case_2():
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

def test_case_3():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__hash__()
    assert var_3 == 3494185624788604514
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_3, position=var_0)

def test_case_4():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = []
    module_0.ParseError(messages=var_0)

def test_case_6():
    var_0 = 'mH{yU>\\t '
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = bool(var_1 == {'field': 'Error'})

def test_case_7():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__bool__()
    assert var_2 is True

def test_case_8():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = bool(var_2 == ['test_value', None])
    assert var_3 is True

def test_case_9():
    var_0 = 'Hw7Y&\x0c@(QgK'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = bool(var_1 == {'field': 'Error'})

def test_case_10():
    var_0 = 'Hw7Y&\x0c@(QgK'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == "{'Hw7Y&\\x0c@(QgK': 'Hw7Y&\\x0c@(QgK'}"

def test_case_11():
    var_0 = []
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_1, messages=var_0)

def test_case_12():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1 == var_1

def test_case_13():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1 == var_1

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 'c&I`,A)y:c6lE\nN'
    var_2 = module_0.Message(text=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c&I`,A)y:c6lE\nN'
    assert var_2.code == 'custom'
    assert var_2.index == ['c&I`,A)y:c6lE\nN']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='c&I`,A)y:c6lE\\nN', code='custom', index=['c&I`,A)y:c6lE\\nN'])"
    var_4 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_2.__hash__()
    assert var_5 == -567444290718348553
    var_6 = var_4.__bool__()
    assert var_6 is True
    module_0.ParseError(text=var_0, code=var_4, key=var_0, position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = set()
    var_5 = module_0.BaseError(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__iter__()
    var_7 = var_5.__contains__(var_1)
    assert var_7 is False
    var_8 = var_5.__contains__(var_7)
    assert var_8 is False
    var_9 = var_5.__eq__(var_0)
    assert var_9 is False
    var_10 = var_5.messages()
    var_7.messages()

def test_case_16():
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
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = True
    var_5 = module_0.Position(var_4, var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert var_5.column_no is True
    assert var_5.char_index is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_3.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_9, messages=var_9)

def test_case_17():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = 'username'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Invalid input'
    assert var_4.code == 'invalid'
    assert var_4.index == ['username']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4, var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = str(var_6)
    var_8 = var_6.__hash__()
    assert var_8 == 8279099173157029692
    var_9 = var_6.__len__()
    assert var_9 == 1
    var_10 = bool(var_7 == var_7)
    assert var_10 is True

def test_case_18():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 4
    var_4 = module_0.Position(var_0, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 4
    assert var_4.char_index == 4
    var_5 = 'Error'
    var_6 = module_0.Message(text=var_5, start_position=var_2, end_position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = module_0.Message(text=var_5, start_position=var_2, end_position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = var_6 == var_7
    assert var_8 is False
    var_9 = var_7.__repr__()
    assert var_9 == "Message(text='Error', code='custom', start_position=Position(line_no=1, column_no=1, char_index=0), end_position=Position(line_no=1, column_no=4, char_index=4))"

def test_case_19():
    var_0 = '\tb|-c.Hy8+&YSqI9@C#'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ValidationError(text='\\tb|-c.Hy8+&YSqI9@C#', code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = False
    var_3 = var_1.__bool__()
    assert var_3 is True
    var_4 = module_0.ParseError(text=var_1, key=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_4) == 1
    var_5 = var_4.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5) == 1
    var_6 = var_5.__eq__(var_0)
    var_7 = var_4.__str__()
    assert var_7 == '{False: ValidationResult(value=None)}'
    var_8 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_5.__repr__()
    assert var_9 == "KeysView(ParseError([Message(text=ValidationResult(value=None), code='custom', index=[False])]))"
    var_10 = module_0.Position(var_5, var_0, var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_10.line_no).__module__}.{type(var_10.line_no).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10.line_no) == 1
    assert var_10.column_no is None
    assert f'{type(var_10.char_index).__module__}.{type(var_10.char_index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10.char_index) == 1
    var_11 = var_1.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_1.__repr__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = var_4.keys()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13) == 1
    module_0.ParseError(messages=var_13)

def test_case_21():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2365
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = True
    var_5 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == -2365
    assert var_5.column_no == -2365
    assert var_5.char_index is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_3.__bool__()
    assert var_7 is True
    var_8 = module_0.ParseError(text=var_1, key=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_1.__eq__(var_0)
    assert var_9 is False
    var_10 = var_8.__str__()
    assert var_10 == "{True: Message(text=None, code='custom')}"
    var_11 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_1.__eq__(var_11)
    assert var_12 is False
    var_13 = var_3.__repr__()
    assert var_13 == 'ValidationResult(value=None)'
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_5, messages=var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2388
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = True
    var_6 = module_0.Position(var_5, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no == -2388
    assert var_6.char_index is True
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = 'F\r2?GQQ{'
    var_9 = module_0.ParseError(text=var_8, key=var_3, messages=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == 'KeysView(ParseError([Message(text=\'F\\r2?GQQ{\', code=\'custom\', index=["Message(text=None, code=\'custom\')"])]))'
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = var_10.__contains__(var_0)
    assert var_13 is False
    var_14 = var_6.__eq__(var_10)
    assert var_14 is False
    var_15 = module_0.ValidationResult(error=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is False
    var_16 = var_15.__repr__()
    assert var_16 == 'ValidationResult(error=False)'
    var_17 = [var_13]
    module_0.BaseError(messages=var_17)

def test_case_23():
    var_0 = 'nested'
    var_1 = 'parent'
    var_2 = '3h&vd'
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_2, code=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '3h&vd'
    assert var_4.code == 'nested'
    assert var_4.index == ['parent', '3h&vd']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = dict(var_6)

def test_case_24():
    var_0 = 'wOCA!3.?)-~8tzw'
    var_1 = 'fild'
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'wOCA!3.?)-~8tzw'
    assert var_2.code == 'wOCA!3.?)-~8tzw'
    assert var_2.index == ['fild']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'wOCA!3.?)-~8tzw'
    assert var_3.code == 'fild'
    assert var_3.index == ['fild']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2 == var_3

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
    var_2 = -2365
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = True
    var_6 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == -2365
    assert var_6.column_no == -2365
    assert var_6.char_index is None
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = var_4.__bool__()
    assert var_8 is True
    var_9 = module_0.ParseError(text=var_1, key=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_1.__eq__(var_0)
    assert var_10 is False
    var_11 = var_9.__str__()
    assert var_11 == "{True: Message(text=None, code='custom')}"
    var_12 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = var_1.__eq__(var_12)
    assert var_13 is False
    var_14 = var_4.__repr__()
    assert var_14 == 'ValidationResult(value=None)'
    var_15 = None
    var_16 = var_1.__eq__(var_15)
    assert var_16 is False
    var_17 = None
    var_18 = var_9.__contains__(var_17)
    assert var_18 is False
    var_19 = var_4.__bool__()
    assert var_19 is True
    module_0.ParseError(code=var_14, position=var_6, messages=var_8)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2388
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = True
    var_5 = module_0.Position(var_4, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert var_5.column_no == -2388
    assert var_5.char_index is True
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = 'F\r2?GQQ{'
    var_8 = module_0.ParseError(text=var_7, key=var_7, messages=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.__len__()
    assert var_9 == 1
    var_10 = var_9.__repr__()
    assert var_10 == '1'
    var_11 = module_0.Message(text=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == '1'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_11.__eq__(var_1)
    assert var_12 is False
    var_13 = var_8.__str__()
    assert var_13 == "{'F\\r2?GQQ{': 'F\\r2?GQQ{'}"
    var_14 = module_0.ValidationResult(value=var_13, error=var_0)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value == "{'F\\r2?GQQ{': 'F\\r2?GQQ{'}"
    assert var_14.error is None
    var_15 = var_8.__contains__(var_3)
    assert var_15 is False
    var_16 = var_1.__eq__(var_15)
    assert var_16 is False
    var_17 = var_3.__repr__()
    assert var_17 == 'ValidationResult(value=None)'
    var_18 = var_8.__len__()
    assert var_18 == 1
    var_19 = var_18.__eq__(var_0)
    var_15.items()

def test_case_27():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = -2365
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = True
    var_7 = module_0.Position(var_3, var_3, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == -2365
    assert var_7.column_no == -2365
    assert var_7.char_index is None
    var_8 = var_7.__eq__(var_0)
    assert var_8 is False
    var_9 = var_2.__hash__()
    assert var_9 == 3494185624788604514
    var_10 = module_0.ParseError(text=var_2, key=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    var_12 = var_2.__eq__(var_1)
    assert var_12 is False
    var_13 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_10.__contains__(var_0)
    assert var_14 is False
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_5, error=var_14)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = -2365
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = module_0.Position(var_3, var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == -2365
    assert var_6.column_no == -2365
    assert var_6.char_index is None
    var_7 = var_5.__bool__()
    assert var_7 is True
    var_8 = var_6.__eq__(var_0)
    assert var_8 is False
    var_9 = var_2.__hash__()
    assert var_9 == 3494185624788604514
    var_10 = False
    var_11 = module_0.ParseError(text=var_2, key=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_11.__str__()
    assert var_12 == "{False: Message(text=None, code='custom')}"
    var_13 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_11.__contains__(var_1)
    assert var_14 is False
    var_15 = var_14.__eq__(var_3)
    assert var_15 is False
    var_16 = var_5.__repr__()
    assert var_16 == 'ValidationResult(value=None)'
    var_17 = var_2.__eq__(var_1)
    assert var_17 is False
    var_18 = var_5.__repr__()
    assert var_18 == 'ValidationResult(value=None)'
    var_19 = None
    var_20 = var_11.keys()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_20) == 1
    var_21 = var_20.__contains__(var_0)
    assert var_21 is False
    var_22 = module_0.BaseError(text=var_3, code=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_22) == 1
    var_23 = var_14.__eq__(var_19)
    var_24 = var_5.__bool__()
    assert var_24 is True
    module_0.ParseError(key=var_20, position=var_0, messages=var_14)

def test_case_29():
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

def test_case_30():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2388
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.__hash__()
    assert var_4 == 3494185624788604514
    var_5 = True
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.Position(var_5, var_2, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no == -2388
    assert var_7.char_index is True
    var_8 = var_7.__eq__(var_0)
    assert var_8 is False
    var_9 = 'F\r2?GQQ{'
    var_10 = module_0.ParseError(text=var_9, key=var_9, messages=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.__len__()
    assert var_11 == 1
    var_12 = var_11.__repr__()
    assert var_12 == '1'
    var_13 = var_1.__eq__(var_0)
    assert var_13 is False
    var_14 = var_10.values()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_14) == 1
    var_15 = var_3.__bool__()
    assert var_15 is True
    var_16 = var_14.__repr__()
    assert var_16 == "ValuesView(ParseError([Message(text='F\\r2?GQQ{', code='custom', index=['F\\r2?GQQ{'])]))"
    var_17 = var_10.messages(add_prefix=var_13)
    var_18 = var_3.__repr__()
    assert var_18 == 'ValidationResult(value=None)'
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_5, messages=var_14)

def test_case_31():
    var_0 = 'Error'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'custom'
    assert var_2.index == ['field']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = [var_1, var_1]
    var_4 = module_0.Message(text=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'custom'
    assert var_4.index == ['field', 'field']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2 == var_4

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = "fw/{PVR=w!NGGu's9k"
    var_2 = module_0.Message(text=var_1, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "fw/{PVR=w!NGGu's9k"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = -2365
    var_4 = var_2.__repr__()
    assert var_4 == 'Message(text="fw/{PVR=w!NGGu\'s9k", code=\'custom\')'
    var_5 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is None
    assert var_6.column_no == -2365
    assert var_6.char_index is None
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = var_2.__hash__()
    assert var_8 == 3494185624788604514
    var_9 = var_5.__bool__()
    assert var_9 is True
    var_10 = module_0.ParseError(text=var_1, messages=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_11) == 1
    var_12 = var_2.__eq__(var_0)
    assert var_12 is False
    var_13 = var_10.__str__()
    assert var_13 == "fw/{PVR=w!NGGu's9k"
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_10.__contains__(var_11)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2365
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == -2365
    assert var_5.column_no == -2365
    assert var_5.char_index is None
    var_6 = True
    var_7 = module_0.Position(var_6, var_0, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is None
    var_8 = var_7.__eq__(var_5)
    assert var_8 is False
    var_9 = var_1.__hash__()
    assert var_9 == 3494185624788604514
    var_10 = var_4.__bool__()
    assert var_10 is True
    module_0.ParseError(text=var_0, key=var_0)

def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ParseError(text=var_1, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__hash__()
    assert var_3 == 5328634574863534199
    var_4 = var_2.__hash__()
    assert var_4 == 5328634574863534199
    var_5 = -2365
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.Position(var_5, var_5, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == -2365
    assert var_8.column_no == -2365
    assert var_8.char_index is None
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = var_1.__hash__()
    assert var_10 == 3494185624788604514
    var_11 = False
    var_12 = var_7.__bool__()
    assert var_12 is True
    var_13 = module_0.ParseError(text=var_1, key=var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = var_1.__eq__(var_0)
    assert var_15 is False
    var_16 = var_13.__str__()
    assert var_16 == "{False: Message(text=None, code='custom')}"
    var_17 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert var_17.error is None
    var_18 = var_13.__contains__(var_0)
    assert var_18 is False
    var_19 = var_18.__repr__()
    assert var_19 == 'False'
    var_20 = var_1.__eq__(var_0)
    assert var_20 is False
    var_21 = var_7.__repr__()
    assert var_21 == 'ValidationResult(value=None)'
    var_22 = None
    var_23 = var_13.keys()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_23) == 1
    var_24 = var_23.__contains__(var_0)
    assert var_24 is False
    var_25 = module_0.BaseError(text=var_16, code=var_16, position=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_25) == 1
    var_26 = var_1.__eq__(var_23)
    assert var_26 is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_22, key=var_24, index=var_18)

def test_case_35():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ParseError(text=var_1, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_4 = -2365
    var_5 = var_2.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_2.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Position(var_4, var_4, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == -2365
    assert var_8.column_no == -2365
    assert var_8.char_index is None
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = var_1.__hash__()
    assert var_10 == 3494185624788604514
    var_11 = False
    var_12 = var_6.__bool__()
    assert var_12 is True
    var_13 = module_0.ParseError(text=var_1, key=var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = var_1.__eq__(var_0)
    assert var_15 is False
    var_16 = var_13.__str__()
    assert var_16 == "{False: Message(text=None, code='custom')}"
    var_17 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert var_17.error is None
    var_18 = var_6.__bool__()
    assert var_18 is True
    var_19 = var_13.__contains__(var_0)
    assert var_19 is False
    var_20 = var_19.__eq__(var_19)
    assert var_20 is True
    var_21 = var_14.__repr__()
    assert var_21 == "KeysView(ParseError([Message(text=Message(text=None, code='custom'), code='custom', index=[False])]))"
    var_22 = var_19.__repr__()
    assert var_22 == 'False'
    var_23 = var_1.__eq__(var_0)
    assert var_23 is False
    var_24 = var_6.__repr__()
    assert var_24 == 'ValidationResult(value=None)'
    var_25 = None
    var_26 = var_13.keys()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_26) == 1
    var_27 = var_26.__contains__(var_0)
    assert var_27 is False
    var_28 = module_0.BaseError(text=var_16, code=var_16, position=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_28) == 1
    var_29 = var_1.__eq__(var_26)
    assert var_29 is False
    var_30 = 'iizr%\x0bYW8w\x0bUf"D|'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_30, index=var_25, position=var_27, start_position=var_2)

def test_case_36():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ParseError(text=var_1, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__hash__()
    assert var_3 == 5328634574863534199
    var_4 = -2352
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_2.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Position(var_4, var_4, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == -2352
    assert var_8.column_no == -2352
    assert var_8.char_index is None
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = var_1.__hash__()
    assert var_10 == 3494185624788604514
    var_11 = False
    var_12 = var_6.__bool__()
    assert var_12 is True
    var_13 = module_0.ParseError(text=var_1, key=var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = var_1.__eq__(var_0)
    assert var_15 is False
    var_16 = var_13.__str__()
    assert var_16 == "{False: Message(text=None, code='custom')}"
    var_17 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert var_17.error is None
    var_18 = var_8.__eq__(var_8)
    assert var_18 is True
    var_19 = var_14.__repr__()
    assert var_19 == "KeysView(ParseError([Message(text=Message(text=None, code='custom'), code='custom', index=[False])]))"
    var_20 = var_10.__repr__()
    assert var_20 == '3494185624788604514'
    var_21 = var_1.__eq__(var_0)
    assert var_21 is False
    var_22 = module_0.Position(var_14, var_0, var_14)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_22.line_no).__module__}.{type(var_22.line_no).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_22.line_no) == 1
    assert var_22.column_no is None
    assert f'{type(var_22.char_index).__module__}.{type(var_22.char_index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_22.char_index) == 1
    var_23 = var_6.__repr__()
    assert var_23 == 'ValidationResult(value=None)'
    var_24 = None
    var_25 = var_13.keys()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_25) == 1
    var_26 = var_25.__contains__(var_0)
    assert var_26 is False
    var_27 = module_0.BaseError(text=var_16, code=var_16, position=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_27) == 1
    var_28 = None
    var_29 = module_0.Message(text=var_14, code=var_28, position=var_26)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_29.text).__module__}.{type(var_29.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_29.text) == 1
    assert var_29.code == 'custom'
    assert var_29.index == []
    assert var_29.start_position is False
    assert var_29.end_position is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_25, index=var_24, position=var_25, end_position=var_5)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ParseError(text=var_1, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__hash__()
    assert var_3 == 5328634574863534199
    var_4 = -2365
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_2.__eq__(var_0)
    assert var_7 is False
    var_8 = var_2.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = module_0.Position(var_4, var_4, var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == -2365
    assert var_9.column_no == -2365
    assert var_9.char_index is None
    var_10 = var_9.__eq__(var_0)
    assert var_10 is False
    var_11 = var_1.__hash__()
    assert var_11 == 3494185624788604514
    var_12 = False
    var_13 = var_6.__bool__()
    assert var_13 is True
    var_14 = module_0.ParseError(text=var_1, key=var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.keys()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_15) == 1
    var_16 = var_2.items()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16) == 1
    var_17 = var_14.__str__()
    assert var_17 == "{False: Message(text=None, code='custom')}"
    var_18 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.value is None
    assert var_18.error is None
    var_19 = var_9.__eq__(var_9)
    assert var_19 is True
    var_20 = var_15.__repr__()
    assert var_20 == "KeysView(ParseError([Message(text=Message(text=None, code='custom'), code='custom', index=[False])]))"
    var_21 = var_11.__repr__()
    assert var_21 == '3494185624788604514'
    var_22 = var_1.__eq__(var_0)
    assert var_22 is False
    var_23 = module_0.Position(var_15, var_0, var_15)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_23.line_no).__module__}.{type(var_23.line_no).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_23.line_no) == 1
    assert var_23.column_no is None
    assert f'{type(var_23.char_index).__module__}.{type(var_23.char_index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_23.char_index) == 1
    var_24 = var_6.__repr__()
    assert var_24 == 'ValidationResult(value=None)'
    var_25 = None
    var_26 = var_14.keys()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_26) == 1
    var_27 = var_26.__contains__(var_0)
    assert var_27 is False
    var_28 = module_0.BaseError(text=var_17, code=var_17, position=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_28) == 1
    var_29 = module_0.Message(text=var_25, code=var_0, start_position=var_27)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Message'
    assert var_29.text is None
    assert var_29.code == 'custom'
    assert var_29.index == []
    assert var_29.start_position is False
    assert var_29.end_position is None
    var_30 = var_1.__eq__(var_29)
    assert var_30 is False
    module_0.ParseError(code=var_0, messages=var_0)

def test_case_38():
    var_0 = 'First error'
    var_1 = 'code1'
    var_2 = 'key1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'First error'
    assert var_4.code == 'code1'
    assert var_4.index == ['key1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Second error'
    var_6 = 'code2'
    var_7 = 'key2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Second error'
    assert var_9.code == 'code2'
    assert var_9.index == ['key2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = repr(var_11)
    var_13 = "BaseError([Message(text='First error', code='code1', index=['key1']), Message(text='Second error', code='code2', index=['key2'])])"
    var_14 = bool(var_12 == var_13)
    assert var_14 is True

def test_case_39():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Invalid input'
    assert var_3.code == 'invalid'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Too short'
    var_5 = 'min_length'
    var_6 = []
    var_7 = module_0.Message(text=var_4, code=var_5, index=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Too short'
    assert var_7.code == 'min_length'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_3, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = str(var_9)
    var_11 = "{'': 'Too short'}"
    var_12 = bool(var_10 == var_11)
    assert var_12 is True

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = '.fx'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '.fx'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = module_0.ValidationError(text=var_0, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_4.__hash__()
    assert var_5 == 1
    var_3.__getitem__(var_2)