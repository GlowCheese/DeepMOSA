# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = '[~vZI[]?Nj*;6'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2.error) == 1

def test_case_1():
    var_0 = 'Error1'
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error1'
    assert var_1.code == 'Error1'
    assert var_1.index == ['Error1']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(not var_1 == var_1)

def test_case_2():
    var_0 = 'KError'
    var_1 = 'a4Yl}vNxw{ '
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'KError'
    assert var_2.code == 'KError'
    assert var_2.index == ['a4Yl}vNxw{ ']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'KError'
    assert var_3.code == 'a4Yl}vNxw{ '
    assert var_3.index == ['a4Yl}vNxw{ ']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_2 == var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = 'W1LpB+C <vXL."\x0b\x0cp'
    var_3 = None
    var_4 = module_0.ParseError(text=var_2, code=var_2, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_1)
    assert var_5 is False
    var_5.__getitem__(var_0)

def test_case_5():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_6():
    var_0 = None
    var_1 = None
    var_2 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__bool__()
    assert var_5 is True
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None

def test_case_7():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = True
    var_3 = None
    var_4 = module_0.Position(var_0, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 'Invalid value'
    assert var_4.column_no is True
    assert var_4.char_index is None
    var_5 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = repr(var_5)
    assert var_6 == "BaseError(text='Invalid value', code='invalid')"

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '[~vZI[]?Nj*;6'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_1.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    module_0.ValidationError(code=var_1, key=var_1)

def test_case_10():
    var_0 = 'Invalid value'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = repr(var_1)

def test_case_11():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = repr(var_3)
    var_5 = var_3.__len__()
    assert var_5 == 1

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = []
    module_0.ValidationError(messages=var_0)

def test_case_13():
    var_0 = True
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 is True
    var_3 = var_1.__iter__()
    var_4 = var_1.__repr__()
    assert var_4 == "ParseError(text=True, code='custom')"
    var_5 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_5)

def test_case_14():
    var_0 = 'Error 1'
    var_1 = 'error1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'error1'
    assert var_4.index == [0]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'error2'
    var_7 = module_0.Message(text=var_5, code=var_6, index=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error 2'
    assert var_7.code == 'error2'
    assert var_7.index == [0]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_4, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = str(var_9)

def test_case_15():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = 'field'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = repr(var_3)

def test_case_16():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is None
    assert var_2.char_index is False
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = module_0.Message(text=var_3, position=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_5.__repr__()
    assert var_7 == "Message(text=None, code='custom', position=Position(line_no=False, column_no=None, char_index=False))"

def test_case_17():
    var_0 = ']L\r{FGRN}N>'
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ']L\r{FGRN}N>'
    assert var_1.code == 'custom'
    assert var_1.index == ']L\r{FGRN}N>'
    assert var_1.start_position == ']L\r{FGRN}N>'
    assert var_1.end_position == ']L\r{FGRN}N>'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_18():
    var_0 = None
    var_1 = None
    var_2 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = var_2.__iter__()
    var_5 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_3.__eq__(var_1)
    var_7 = var_5.__eq__(var_1)
    assert var_7 is False
    var_8 = None
    var_9 = var_5.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_6, code=var_8, messages=var_6)

def test_case_19():
    var_0 = 'test'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test'
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = bool(var_2 == ['test', None])
    assert var_3 is True

def test_case_20():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = str(var_1)
    assert var_2 == 'Error message'

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 'WMSCS+kTL/]>5n)kYW'
    module_0.ParseError(code=var_0, messages=var_0)

def test_case_22():
    var_0 = 'UBHr8BK-LJ1*66i=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = var_4.__iter__()
    var_7 = var_2.__contains__(var_0)
    assert var_7 is False
    var_8 = module_0.Message(text=var_5, key=var_5, position=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'ValidationResult(value=None)'
    assert var_8.code == 'custom'
    assert var_8.index == ['ValidationResult(value=None)']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_4.__bool__()
    assert var_9 is True
    var_10 = None
    var_11 = var_8.__repr__()
    assert var_11 == "Message(text='ValidationResult(value=None)', code='custom', index=['ValidationResult(value=None)'])"
    var_12 = module_0.BaseError(text=var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    var_13 = var_12.__contains__(var_3)
    assert var_13 is False
    var_14 = var_12.__eq__(var_10)
    assert var_14 is False
    var_15 = var_2.__len__()
    assert var_15 == 1
    var_16 = module_0.Message(text=var_13, start_position=var_7, end_position=var_3)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is False
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position is False
    assert var_16.end_position is None
    var_17 = 'GsZWDV|N<q3A\x0c{H94|'
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_17, messages=var_15)

def test_case_23():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = module_0.ValidationError(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.value) == 1
    assert var_3.error is None
    var_4 = var_1.messages(add_prefix=var_2)
    var_5 = var_3.__repr__()
    assert var_5 == "ValidationResult(value=ValidationError([Message(text='XBHr8BK-LJ1*66=<2', code='XBHr8BK-LJ1*66=<2', index=['XBHr8BK-LJ1*66=<2'])]))"
    var_6 = module_0.Message(text=var_0, index=var_2, position=var_2, end_position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'XBHr8BK-LJ1*66=<2'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__bool__()
    assert var_7 is True
    var_8 = var_3.__bool__()
    assert var_8 is True
    var_9 = var_1.__eq__(var_2)
    assert var_9 is False
    var_10 = var_1.__str__()
    assert var_10 == "{'XBHr8BK-LJ1*66=<2': 'XBHr8BK-LJ1*66=<2'}"
    var_11 = var_1.__iter__()
    var_12 = var_1.values()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_12) == 1

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 'UBHr8BK-LJ1*66i=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = None
    var_5 = None
    var_6 = module_0.ValidationResult(value=var_5, error=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_6.__iter__()
    var_9 = var_6.__iter__()
    var_10 = 'X3X<'
    var_11 = module_0.Message(text=var_10, key=var_5, position=var_3, start_position=var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'X3X<'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position == 'ValidationResult(value=None)'
    assert var_11.end_position is None
    var_12 = var_6.__bool__()
    assert var_12 is True
    var_13 = var_11.__eq__(var_5)
    assert var_13 is False
    var_14 = var_11.__repr__()
    assert var_14 == "Message(text='X3X<', code='custom', start_position='ValidationResult(value=None)', end_position=None)"
    var_15 = module_0.BaseError(text=var_7)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = var_2.__contains__(var_5)
    assert var_16 is False
    var_17 = var_15.__eq__(var_16)
    assert var_17 is False
    var_18 = var_15.__len__()
    assert var_18 == 1
    var_19 = var_15.__eq__(var_5)
    assert var_19 is False
    var_16.__iter__()

def test_case_25():
    var_0 = 'UBHr8BK-LJ1*66i=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.value) == 1
    assert var_3.error is None
    var_4 = var_3.__iter__()
    var_5 = '>tv o%^\t:\x0coa"Y\\9'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, key=var_4, index=var_0)

def test_case_26():
    var_0 = None
    var_1 = False
    var_2 = False
    var_3 = module_0.Position(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is None
    assert var_3.char_index is False
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_3.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = False
    var_4 = False
    var_5 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is False
    var_6 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.Position'
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(error=Position(line_no=False, column_no=None, char_index=False))'
    var_8 = var_5.__eq__(var_1)
    assert var_8 is False
    var_9 = None
    var_10 = var_2.messages(add_prefix=var_1)
    var_11 = module_0.Message(text=var_9, key=var_9, position=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is None
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_11.__eq__(var_1)
    assert var_12 is False
    var_13 = None
    var_14 = var_2.__eq__(var_9)
    assert var_14 is False
    var_15 = var_11.__repr__()
    assert var_15 == "Message(text=None, code='custom')"
    var_16 = var_2.__str__()
    assert var_16 == 'XBHr8BK-LJ1*66=<2'
    var_2.__getitem__(var_13)

def test_case_29():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = module_0.ValidationError(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.value) == 1
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == "ValidationResult(value=ValidationError([Message(text='XBHr8BK-LJ1*66=<2', code='XBHr8BK-LJ1*66=<2', index=['XBHr8BK-LJ1*66=<2'])]))"
    var_5 = False
    var_6 = 331
    var_7 = module_0.Position(var_6, var_5, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 331
    assert var_7.column_no is False
    assert var_7.char_index is None
    var_8 = var_4.__len__()
    assert var_8 == 131
    var_9 = var_1.__len__()
    assert var_9 == 1
    var_10 = var_3.__iter__()
    var_11 = var_7.__repr__()
    assert var_11 == 'Position(line_no=331, column_no=False, char_index=None)'
    var_12 = var_9.__eq__(var_2)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_4, error=var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = True
    var_4 = False
    var_5 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert var_5.column_no is None
    assert var_5.char_index is False
    var_6 = var_5.__eq__(var_1)
    assert var_6 is False
    var_7 = None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = -764.84
    var_10 = var_2.messages(add_prefix=var_9)
    var_11 = var_8.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_8.__iter__()
    var_13 = var_2.__iter__()
    var_14 = var_8.__bool__()
    assert var_14 is True
    var_15 = var_13.__eq__(var_7)
    var_16 = var_2.__eq__(var_1)
    assert var_16 is False
    var_17 = var_5.__repr__()
    assert var_17 == 'Position(line_no=True, column_no=None, char_index=False)'
    var_18 = var_2.__str__()
    assert var_18 == 'XBHr8BK-LJ1*66=<2'
    var_2.__getitem__(var_18)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = 2507
    var_4 = module_0.Position(var_1, var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index == 2507
    var_5 = False
    var_6 = False
    var_7 = module_0.Position(var_5, var_1, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is False
    assert var_7.column_no is None
    assert var_7.char_index is False
    var_8 = var_7.__eq__(var_1)
    assert var_8 is False
    var_9 = var_2.__contains__(var_6)
    assert var_9 is False
    var_10 = var_2.messages(add_prefix=var_1)
    var_11 = var_9.__repr__()
    assert var_11 == 'False'
    var_12 = var_7.__eq__(var_4)
    assert var_12 is False
    var_9.__iter__()

def test_case_32():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationError(text=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = False
    var_5 = False
    var_6 = False
    var_7 = module_0.Position(var_5, var_4, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is False
    assert var_7.column_no is False
    assert var_7.char_index is False
    var_8 = -501
    var_9 = module_0.Position(var_1, var_1, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no is None
    assert var_9.char_index == -501
    var_10 = var_9.__eq__(var_1)
    assert var_10 is False
    var_11 = module_0.ValidationResult(error=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert f'{type(var_11.error).__module__}.{type(var_11.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11.error) == 1
    var_12 = var_3.messages(add_prefix=var_1)
    var_13 = var_11.__repr__()
    assert var_13 == "ValidationResult(error=ValidationError(text='XBHr8BK-LJ1*66=<2', code='custom'))"
    var_14 = var_9.__eq__(var_1)
    assert var_14 is False
    var_15 = var_11.__iter__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_12, position=var_7, end_position=var_9)

def test_case_33():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = False
    var_4 = False
    var_5 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is False
    var_6 = var_5.__eq__(var_1)
    assert var_6 is False
    var_7 = var_2.values()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_7) == 1
    var_8 = var_2.messages()
    var_9 = var_7.__repr__()
    assert var_9 == "ValuesView(ValidationError(text='XBHr8BK-LJ1*66=<2', code='custom'))"
    var_10 = var_5.__eq__(var_1)
    assert var_10 is False
    var_11 = var_2.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    var_12 = var_11.__iter__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_7, code=var_11, position=var_7, start_position=var_10, end_position=var_7)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = False
    var_4 = False
    var_5 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is None
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is False
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_1)
    assert var_7 is False
    var_8 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.value) == 1
    assert var_8.error is None
    var_9 = var_2.messages(add_prefix=var_1)
    var_10 = var_8.__repr__()
    assert var_10 == "ValidationResult(value=ValidationError(text='XBHr8BK-LJ1*66=<2', code='custom'))"
    var_11 = var_6.__eq__(var_1)
    assert var_11 is False
    var_12 = module_0.Message(text=var_1, position=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text is None
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert f'{type(var_12.start_position).__module__}.{type(var_12.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_12.end_position).__module__}.{type(var_12.end_position).__qualname__}' == 'typesystem.base.Position'
    var_13 = var_8.__bool__()
    assert var_13 is True
    var_14 = var_12.__eq__(var_1)
    assert var_14 is False
    var_15 = var_6.__repr__()
    assert var_15 == 'Position(line_no=False, column_no=None, char_index=False)'
    var_16 = None
    var_17 = None
    var_18 = var_2.__eq__(var_16)
    assert var_18 is False
    var_19 = var_12.__repr__()
    assert var_19 == "Message(text=None, code='custom', position=Position(line_no=False, column_no=None, char_index=False))"
    var_20 = var_2.__str__()
    assert var_20 == 'XBHr8BK-LJ1*66=<2'
    var_21 = var_2.__iter__()
    module_0.ValidationError(key=var_17, position=var_5, messages=var_9)

def test_case_35():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = module_0.Position(var_1, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert f'{type(var_4.column_no).__module__}.{type(var_4.column_no).__qualname__}' == 'typesystem.base.Position'
    assert var_4.char_index is None
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_2.messages(add_prefix=var_0)
    var_8 = var_6.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_4.__eq__(var_3)
    assert var_9 is False
    var_10 = module_0.Message(text=var_1, code=var_7, start_position=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert f'{type(var_10.code).__module__}.{type(var_10.code).__qualname__}' == 'builtins.list'
    assert len(var_10.code) == 1
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_6.__bool__()
    assert var_11 is True
    var_12 = var_2.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_12.__eq__(var_1)
    var_14 = var_12.__eq__(var_12)
    assert var_14 is True
    var_15 = var_10.__repr__()
    assert var_15 == "Message(text=None, code=[Message(text='XBHr8BK-LJ1*66=<2', code='custom', index=['XBHr8BK-LJ1*66=<2'])])"
    var_16 = var_2.__str__()
    assert var_16 == 'XBHr8BK-LJ1*66=<2'
    var_17 = var_12.__iter__()
    var_18 = module_0.ValidationError(text=var_15)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18) == 1
    var_19 = var_18.__iter__()
    var_20 = var_12.__iter__()
    var_21 = var_3.__eq__(var_0)
    assert var_21 is False

def test_case_36():
    var_0 = 'XBHr8BK-LJ1*66=<2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = False
    var_4 = False
    var_5 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is None
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = module_0.Position(var_3, var_1, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is False
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_1)
    assert var_7 is False
    var_8 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.value) == 1
    assert var_8.error is None
    var_9 = var_2.messages()
    var_10 = var_8.__repr__()
    assert var_10 == "ValidationResult(value=ValidationError(text='XBHr8BK-LJ1*66=<2', code='custom'))"
    var_11 = var_6.__eq__(var_1)
    assert var_11 is False
    var_12 = var_8.__iter__()
    var_13 = False
    var_14 = [var_4, var_11, var_13, var_10]
    var_15 = module_0.Message(text=var_0, code=var_1, key=var_1, index=var_14, position=var_6, end_position=var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'XBHr8BK-LJ1*66=<2'
    assert var_15.code == 'custom'
    assert var_15.index == [False, False, False, "ValidationResult(value=ValidationError(text='XBHr8BK-LJ1*66=<2', code='custom'))"]
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.Position'
    var_16 = var_8.__bool__()
    assert var_16 is True
    var_17 = var_15.__eq__(var_15)
    assert var_17 is True
    var_18 = None
    var_19 = var_2.__eq__(var_18)
    assert var_19 is False
    var_20 = None
    var_21 = var_15.__repr__()
    assert var_21 == 'Message(text=\'XBHr8BK-LJ1*66=<2\', code=\'custom\', index=[False, False, False, "ValidationResult(value=ValidationError(text=\'XBHr8BK-LJ1*66=<2\', code=\'custom\'))"], position=Position(line_no=False, column_no=None, char_index=False))'
    var_22 = var_2.__str__()
    assert var_22 == 'XBHr8BK-LJ1*66=<2'
    var_23 = var_2.__iter__()
    var_24 = [var_15]
    var_25 = module_0.ValidationError(key=var_1, position=var_20, messages=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_25) == 1
    var_26 = var_25.__iter__()
    var_27 = var_25.__iter__()
    var_28 = var_25.__eq__(var_20)
    assert var_28 is False
    var_29 = var_28.__eq__(var_18)

def test_case_37():
    var_0 = 'tKest'
    var_1 = 'field'
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'tKest'
    assert var_2.code == 'tKest'
    assert var_2.index == ['field']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_1, code=var_0, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'field'
    assert var_3.code == 'tKest'
    assert var_3.index == ['field']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_2 == var_3)

def test_case_38():
    var_0 = 'Error'
    var_1 = 'fied'
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'Error'
    assert var_2.index == ['fied']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'test2'
    var_4 = module_0.Message(text=var_0, code=var_3, key=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'test2'
    assert var_4.index == ['fied']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = bool(not var_2 == var_4)
    assert var_5 is True

def test_case_39():
    var_0 = 'test'
    var_1 = 'a'
    var_2 = 1
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_1, code=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'a'
    assert var_4.code == 'test'
    assert var_4.index == ['a', 1]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'b'
    var_6 = 2
    var_7 = [var_5, var_6]
    var_8 = module_0.Message(text=var_1, code=var_0, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'a'
    assert var_8.code == 'test'
    assert var_8.index == ['b', 2]
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = bool(not var_4 == var_8)
    assert var_9 is True

def test_case_40():
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
    var_12 = repr(var_11)