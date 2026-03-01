# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 736
    var_1 = None
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 736
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = False
    var_4 = module_0.Position(var_0, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 736
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_2.__eq__(var_1)
    assert var_5 is False
    var_6 = var_4.__eq__(var_2)
    assert var_6 is False

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_2():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_3():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_0.__repr__()
    assert var_2 == 'None'
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.ValidationError(text=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    module_0.BaseError(text=var_0, code=var_0, position=var_0, messages=var_2)

def test_case_6():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None

def test_case_7():
    var_0 = ''
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = var_2.__repr__()
    assert var_3 == '1'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_4.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = None
    module_0.ParseError(messages=var_2)

def test_case_10():
    var_0 = 'test_error'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = ''
    module_0.ValidationError(text=var_0, code=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.__str__()
    assert var_5 is True
    var_4.__getitem__(var_2)

def test_case_13():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = None
    var_3 = '-'
    var_4 = module_0.Message(text=var_3, start_position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '-'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__hash__()
    assert var_5 == -7378205455028346897

def test_case_14():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = '~?&WKy$Fxa"$S7-,3i'
    var_3 = module_0.ParseError(text=var_2, key=var_2, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = var_1.__eq__(var_0)
    assert var_5 is False
    var_6 = var_3.__iter__()

def test_case_15():
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
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

def test_case_16():
    var_0 = ''
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "KeysView(ValidationError(text='', code='custom'))"

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'test_error'
    var_1 = None
    var_2 = module_0.Message(text=var_1, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == ['test_error']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    module_0.ValidationError(messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = 'test_error'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_0, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'test_error'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    module_0.ValidationError(text=var_1, code=var_1, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_4.__getitem__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = 736
    var_1 = None
    var_2 = -768
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -768
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=736, column_no=None, char_index=-768)'
    var_6 = None
    var_7 = var_4.__repr__()
    assert var_7 == 'False'
    module_0.ValidationError(text=var_6, code=var_6, messages=var_6)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 736
    var_1 = None
    var_2 = -768
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -768
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = None
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=736, column_no=None, char_index=-768)'
    var_7 = module_0.Message(text=var_5, end_position=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_7.__eq__(var_1)
    assert var_9 is False
    var_10 = var_7.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = module_0.BaseError(text=var_6, code=var_1, key=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_7.__repr__()
    assert var_12 == "Message(text=None, code='custom')"
    var_13 = []
    module_0.ValidationError(messages=var_13)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.BaseError(text=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_5.__len__()
    assert var_7 == 1
    var_8 = var_6.__repr__()
    assert var_8 == "[Message(text=False, code='custom')]"
    var_9 = var_5.__iter__()
    var_10 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_9.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    module_0.ValidationError(messages=var_0)

def test_case_24():
    var_0 = None
    var_1 = 'h.m'
    var_2 = module_0.Message(text=var_1, key=var_1, index=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'h.m'
    assert var_2.code == 'custom'
    assert var_2.index == ['h.m']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='h.m', code='custom', index=['h.m'])"
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_1, key=var_0)

def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = var_1.__eq__(var_2)
    assert var_4 is False
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_0, key=var_4, messages=var_5)

def test_case_26():
    var_0 = 'test_error'
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test_error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == 'test_error'
    assert var_1.end_position == 'test_error'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text='test_error', code='custom', position='test_error')"
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text='test_error', code='custom', position='test_error')"

def test_case_27():
    var_0 = 'test_error'
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test_error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == 'test_error'
    assert var_1.end_position == 'test_error'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_28():
    var_0 = None
    var_1 = -768
    var_2 = module_0.Position(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == -768
    assert var_2.column_no is None
    assert var_2.char_index == -768
    var_3 = 'z+qQvF'
    var_4 = '`xc{SXNic'
    var_5 = module_0.Message(text=var_3, index=var_0, start_position=var_4, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'z+qQvF'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position == '`xc{SXNic'
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_5.__repr__()
    assert var_7 == "Message(text='z+qQvF', code='custom', start_position='`xc{SXNic', end_position=None)"
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 736
    var_1 = None
    var_2 = -768
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -768
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=736, column_no=None, char_index=-768)'
    var_6 = var_3.__eq__(var_3)
    assert var_6 is True
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    module_0.ValidationError(messages=var_1)

def test_case_30():
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

def test_case_31():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.BaseError(text=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_5.__len__()
    assert var_7 == 1
    var_8 = var_6.__repr__()
    assert var_8 == "[Message(text=False, code='custom')]"
    var_9 = var_5.__iter__()
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_8, error=var_9)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = 'h.m'
    var_2 = module_0.Message(text=var_1, index=var_1, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'h.m'
    assert var_2.code == 'custom'
    assert var_2.index == 'h.m'
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='h.m', code='custom', index='h.m')"
    var_5 = module_0.BaseError(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_5.__getitem__(var_3)

def test_case_33():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = '~?&WKy$Fxa"$S7-,i'
    var_3 = module_0.ParseError(text=var_2, key=var_2, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = var_1.__eq__(var_0)
    assert var_5 is False

def test_case_34():
    var_0 = 736
    var_1 = None
    var_2 = -768
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -768
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = None
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=736, column_no=None, char_index=-768)'
    var_7 = module_0.Message(text=var_5, end_position=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_7.__eq__(var_1)
    assert var_9 is False
    var_10 = var_7.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = [var_7, var_7, var_7, var_7]
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_3, messages=var_11)

def test_case_35():
    var_0 = 736
    var_1 = None
    var_2 = -768
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == -768
    assert var_3.column_no == -768
    assert var_3.char_index == 736
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = None
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=-768, column_no=-768, char_index=736)'
    var_7 = False
    var_8 = module_0.Position(var_5, var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is False
    assert var_8.char_index is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, position=var_8, start_position=var_8, end_position=var_8)

def test_case_36():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = var_0.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = -817
    var_2 = module_0.Position(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == -817
    assert var_2.column_no is None
    assert var_2.char_index == -817
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = None
    var_5 = var_2.__repr__()
    assert var_5 == 'Position(line_no=-817, column_no=None, char_index=-817)'
    var_6 = module_0.Message(text=var_4, end_position=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_6.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = module_0.Position(var_0, var_4, var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is None
    assert var_10.char_index is False
    var_11 = module_0.ValidationError(text=var_9, position=var_10, messages=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    module_0.ValidationError(code=var_1, messages=var_3)

def test_case_38():
    var_0 = 736
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)

def test_case_39():
    var_0 = 736
    var_1 = None
    var_2 = -1400
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -1400
    var_4 = None
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=736, column_no=None, char_index=-1400)'
    var_6 = module_0.Message(text=var_4, end_position=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_3.__eq__(var_3)
    assert var_8 is True
    var_9 = var_6.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_6.__hash__()
    assert var_10 == -7378205455028346897
    var_11 = ';BlLD:a*ao2"^e'
    var_12 = module_0.ParseError(text=var_5, key=var_11, position=var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_12.__str__()
    assert var_13 == '{\';BlLD:a*ao2"^e\': \'Position(line_no=736, column_no=None, char_index=-1400)\'}'

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 736
    var_1 = None
    var_2 = -1400
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -1400
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = None
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=736, column_no=None, char_index=-1400)'
    var_7 = module_0.Message(text=var_5, end_position=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_3.__eq__(var_3)
    assert var_9 is True
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_7.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = var_7.__hash__()
    assert var_12 == -7378205455028346897
    var_13 = module_0.BaseError(text=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = module_0.Position(var_12, var_4, var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no == -7378205455028346897
    assert var_14.column_no is False
    assert var_14.char_index == -7378205455028346897
    var_15 = var_13.messages(add_prefix=var_12)
    var_16 = var_13.values()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_16) == 1
    module_0.ParseError(code=var_1, key=var_4, position=var_5)

def test_case_41():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'field'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='Error message', code='error_code', index=['field'])])"
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
    var_11 = 'code2'
    var_12 = 'field2'
    var_13 = module_0.Message(text=var_7, code=var_11, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'Error 1'
    assert var_13.code == 'code2'
    assert var_13.index == ['field2']
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = [var_10, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 2
    var_16 = repr(var_15)
    var_17 = 22
    var_18 = 3
    var_19 = module_0.Position(var_17, var_3, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no == 22
    assert var_19.column_no == "BaseError(text='Error message', code='error_code')"
    assert var_19.char_index == 3
    var_20 = module_0.BaseError(text=var_0, code=var_1, position=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = repr(var_20)

def test_case_42():
    var_0 = 736
    var_1 = None
    var_2 = -1400
    var_3 = module_0.Position(var_1, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no == -1400
    assert var_3.char_index == -1400
    var_4 = None
    var_5 = var_3.__repr__()
    assert var_5 == 'Position(line_no=None, column_no=-1400, char_index=-1400)'
    var_6 = 'tI4"@ukusWmV|<J6n_'
    var_7 = module_0.Message(text=var_6, key=var_1, start_position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'tI4"@ukusWmV|<J6n_'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_1)
    assert var_8 is False
    var_9 = 1239
    var_10 = module_0.Position(var_0, var_2, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 736
    assert var_10.column_no == -1400
    assert var_10.char_index == 1239
    var_11 = var_3.__eq__(var_10)
    assert var_11 is False
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = var_7.__repr__()
    assert var_13 == 'Message(text=\'tI4"@ukusWmV|<J6n_\', code=\'custom\')'
    var_14 = var_7.__hash__()
    assert var_14 == -7378205455028346897
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_8, position=var_4)

def test_case_43():
    var_0 = 736
    var_1 = None
    var_2 = -1400
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -1400
    var_4 = '703d\\,t^'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_4, index=var_1, position=var_3, end_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 736
    var_1 = None
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 736
    assert var_2.column_no is None
    assert var_2.char_index == 736
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=736, column_no=None, char_index=736)'
    var_5 = var_2.__eq__(var_1)
    assert var_5 is False
    var_6 = var_2.__repr__()
    assert var_6 == 'Position(line_no=736, column_no=None, char_index=736)'
    var_7 = module_0.Message(text=var_0, key=var_1, index=var_1, start_position=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 736
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_1)
    assert var_8 is False
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = module_0.ValidationResult(error=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is False
    var_11 = var_7.__repr__()
    assert var_11 == "Message(text=736, code='custom')"
    var_12 = var_10.__bool__()
    assert var_12 is False
    var_13 = module_0.BaseError(text=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_13.messages()
    var_15 = var_10.__repr__()
    assert var_15 == 'ValidationResult(error=False)'
    var_16 = var_13.__str__()
    assert var_16 == 'Position(line_no=736, column_no=None, char_index=736)'
    var_13.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = 736
    var_1 = None
    var_2 = -1362
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 736
    assert var_3.column_no is None
    assert var_3.char_index == -1362
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=736, column_no=None, char_index=-1362)'
    var_7 = module_0.Message(text=var_1, end_position=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_3.__eq__(var_3)
    assert var_9 is True
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_7.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = var_10.__bool__()
    assert var_12 is True
    var_13 = module_0.BaseError(text=var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_13.messages(add_prefix=var_4)
    var_15 = var_10.__repr__()
    assert var_15 == 'ValidationResult(value=None)'
    var_16 = var_13.__str__()
    assert var_16 == 'Position(line_no=736, column_no=None, char_index=-1362)'
    var_17 = var_13.__str__()
    assert var_17 == 'Position(line_no=736, column_no=None, char_index=-1362)'
    var_18 = var_13.values()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_18) == 1
    var_19 = var_18.__str__()
    assert var_19 == "ValuesView(BaseError(text='Position(line_no=736, column_no=None, char_index=-1362)', code='custom'))"
    var_20 = module_0.ValidationError(code=var_1, messages=var_14)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20) == 1
    var_21 = var_10.__repr__()
    assert var_21 == 'ValidationResult(value=None)'
    var_22 = var_3.__eq__(var_13)
    assert var_22 is False
    var_23 = var_13.__eq__(var_20)
    assert var_23 is False
    var_23.__getitem__(var_18)

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = None
    var_1 = -1400
    var_2 = module_0.Position(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == -1400
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=-1400, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_2.__eq__(var_3)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_5.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_8.__bool__()
    assert var_10 is True
    var_11 = [var_5, var_5]
    var_12 = module_0.BaseError(text=var_0, code=var_0, position=var_0, messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    var_13 = var_12.messages()
    var_14 = var_8.__repr__()
    assert var_14 == 'ValidationResult(value=None)'
    var_15 = var_12.__str__()
    assert var_15 == "{'': None}"
    var_16 = var_12.__iter__()
    var_17 = var_16.__str__()
    var_16.values()

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = 'tQst_err1Zr'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_0, key=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'tQst_err1Zr'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 8366166758319301230
    var_4 = module_0.Message(text=var_0, key=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'tQst_err1Zr'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_2)
    assert var_5 is False
    var_6 = module_0.ValidationError(text=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_6.__iter__()
    var_7.__contains__(var_1)

def test_case_48():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)
    assert var_3 == "BaseError(text='Error message', code='error_code')"
    var_4 = 'field'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = repr(var_5)
    assert var_6 == "BaseError([Message(text='Error message', code='error_code', index=['field'])])"
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
    var_11 = 'code2'
    var_12 = 'field2'
    var_13 = module_0.Message(text=var_7, code=var_11, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'Error 1'
    assert var_13.code == 'code2'
    assert var_13.index == ['field2']
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = [var_10, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 2
    var_16 = repr(var_15)
    var_17 = 22
    var_18 = 3
    var_19 = module_0.Position(var_17, var_3, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no == 22
    assert var_19.column_no == "BaseError(text='Error message', code='error_code')"
    assert var_19.char_index == 3
    var_20 = module_0.BaseError(text=var_0, code=var_1, position=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = var_10.__eq__(var_13)
    assert var_21 is False
    var_22 = repr(var_20)

def test_case_49():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3._messages
    var_5 = len(var_4)
    assert var_5 == 1
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
    var_12 = 'Message 1'
    var_13 = 'code1'
    var_14 = 'key1'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'Message 1'
    assert var_15.code == 'code1'
    assert var_15.index == ['key1']
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = 'Message 2'
    var_17 = 'code2'
    var_18 = 'nested'
    var_19 = 'key'
    var_20 = [var_18, var_19]
    var_21 = module_0.Message(text=var_16, code=var_17, index=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text == 'Message 2'
    assert var_21.code == 'code2'
    assert var_21.index == ['nested', 'key']
    assert var_21.start_position is None
    assert var_21.end_position is None
    var_22 = [var_15, var_21]
    var_23 = module_0.BaseError(messages=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_23) == 2
    var_24 = var_23._messages
    var_25 = len(var_24)
    assert var_25 == 2
    var_26 = 'prefix'
    var_27 = var_23.messages(add_prefix=var_26)
    var_28 = dict(var_3)
    var_29 = list(var_3)
    var_30 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_30) == 1
    var_31 = repr(var_3)

def test_case_50():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
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
    var_4 = module_0.Message(text=var_0, index=var_0, start_position=var_3, end_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is False
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_7)