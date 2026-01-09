# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0


def test_case_0():
    var_0 = None
    var_1 = '2eJC'
    var_2 = module_0.ValidationError(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1

def test_case_1():
    var_0 = None
    var_1 = '2eJCV'
    var_2 = module_0.Message(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '2eJCV'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False

def test_case_2():
    var_0 = None
    var_1 = '2eJCV'
    var_2 = module_0.Message(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '2eJCV'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='2eJCV', code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = '2eJCV'
    var_2 = var_1.__eq__(var_0)
    var_3 = module_0.ValidationError(text=var_1, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    var_5 = module_0.Message(text=var_4, position=var_0, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_5.text) == 1
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    module_0.BaseError(messages=var_4)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.ParseError()

def test_case_5():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_6():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

def test_case_7():
    var_0 = None
    var_1 = 1394
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == 1394
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Message(text=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == [False]
    assert var_4.start_position is None
    assert var_4.end_position is None

def test_case_8():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 1394
    var_4 = module_0.Position(var_3, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1394
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_4.__repr__()
    assert var_7 == 'Position(line_no=1394, column_no=None, char_index=None)'
    var_8 = var_1.__repr__()
    assert var_8 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = 'h[y'
    var_2 = None
    var_3 = module_0.ParseError(text=var_1, key=var_2, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_3.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = None
    module_0.ValidationError(text=var_2, key=var_2, position=var_2)

def test_case_11():
    var_0 = 'invalid_json'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'invalid_json'
    assert var_2.code == 'invalid_json'
    assert var_2.index == ['invalid_json']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'Missing fieC"]ld'
    var_4 = 'missing_field'
    var_5 = 'field2'
    var_6 = module_0.Message(text=var_3, code=var_4, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Missing fieC"]ld'
    assert var_6.code == 'missing_field'
    assert var_6.index == ['field2']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = [var_2, var_6]
    var_8 = module_0.ParseError(messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 2
    var_9 = 'Invalid value'
    var_10 = None
    var_11 = var_6.__eq__(var_10)
    assert var_11 is False
    var_12 = 'subfield'
    var_13 = [var_3, var_12]
    var_14 = module_0.Message(text=var_9, code=var_9, index=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'Invalid value'
    assert var_14.code == 'Invalid value'
    assert var_14.index == ['Missing fieC"]ld', 'subfield']
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = 'missing'
    var_16 = [var_5]
    var_17 = module_0.Message(text=var_3, code=var_15, index=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'Missing fieC"]ld'
    assert var_17.code == 'missing'
    assert var_17.index == ['field2']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = 'All wtests passed!'
    var_19 = print(var_18)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '4de3BEs&jy;'
    var_1 = None
    var_2 = 'h[y'
    var_3 = module_0.BaseError(text=var_2, code=var_0, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__hash__()
    assert var_4 == -4702653217109721937
    var_5 = var_3.__repr__()
    assert var_5 == "BaseError(text='h[y', code='4de3BEs&jy;')"
    var_6 = var_3.__repr__()
    assert var_6 == "BaseError(text='h[y', code='4de3BEs&jy;')"
    module_0.ParseError(messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    var_1 = '2eJCV'
    var_2 = module_0.Message(text=var_1, index=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '2eJCV'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='2eJCV', code='custom')"
    module_0.ParseError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '2eJCV'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = '2eJCV'
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == ['2eJCV']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='custom', index=['2eJCV'])"
    module_0.ValidationError(code=var_0, position=var_0, messages=var_0)

def test_case_16():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = 'MNO/big"glIpF{('
    var_3 = module_0.Message(text=var_2, key=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'MNO/big"glIpF{('
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == 'Message(text=\'MNO/big"glIpF{(\', code=\'custom\')'
    var_6 = module_0.ValidationError(text=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_6.values()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_7) == 1
    var_8 = 'tP'
    var_9 = module_0.Message(text=var_2, key=var_8, start_position=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'MNO/big"glIpF{('
    assert var_9.code == 'custom'
    assert var_9.index == ['tP']
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_9.end_position is None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_7, code=var_0, messages=var_7)

def test_case_17():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = '4de3BEs&jy;'
    var_4 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value == '4de3BEs&jy;'
    assert var_4.error is None
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_18():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, position=var_1, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_2.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.ValidationError(text=var_4, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = '2eJCV'
    var_2 = module_0.Message(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '2eJCV'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='2eJCV', code='custom')"
    var_5 = None
    var_6 = [var_2, var_2]
    var_7 = module_0.ParseError(key=var_5, messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_7.__repr__()
    assert var_8 == "ParseError([Message(text='2eJCV', code='custom'), Message(text='2eJCV', code='custom')])"
    var_9 = var_8.__hash__()
    assert var_9 == -5721415646360035525
    var_10 = var_8.__eq__(var_5)
    var_11 = var_2.__repr__()
    assert var_11 == "Message(text='2eJCV', code='custom')"
    module_0.ParseError(text=var_0)

def test_case_20():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_4 = 'P#;2 U<T?'
    var_5 = 'm/A"+\x0b)>rT~jf,'
    var_6 = module_0.Message(text=var_4, key=var_5, position=var_1, end_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'P#;2 U<T?'
    assert var_6.code == 'custom'
    assert var_6.index == ['m/A"+\x0b)>rT~jf,']
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = var_6.__repr__()
    assert var_7 == 'Message(text=\'P#;2 U<T?\', code=\'custom\', index=[\'m/A"+\\x0b)>rT~jf,\'], position=Position(line_no=None, column_no=None, char_index=None))'

def test_case_21():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, position=var_1, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = 'z(p>f[yC4'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_7, position=var_1, end_position=var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, key=var_0, index=var_0, position=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = None
    var_9 = var_2.__hash__()
    assert var_9 == 7523839687821003752
    var_10 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_10.value).__module__}.{type(var_10.value).__qualname__}' == 'typesystem.base.Position'
    assert var_10.error is None
    var_11 = module_0.Message(text=var_0, index=var_0, start_position=var_7, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is None
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position == "Message(text=None, code='custom')"
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.ValidationResult'
    var_12 = var_11.__eq__(var_4)
    assert var_12 is False
    var_13 = module_0.Position(var_8, var_8, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no is None
    assert var_13.char_index is None
    var_14 = var_11.__repr__()
    assert var_14 == 'Message(text=None, code=\'custom\', start_position="Message(text=None, code=\'custom\')", end_position=ValidationResult(value=Position(line_no=None, column_no=None, char_index=None)))'
    module_0.ParseError(position=var_8)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, key=var_0, index=var_0, position=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = '4de3BEs&jy;'
    var_9 = var_2.__hash__()
    assert var_9 == 7523839687821003752
    var_10 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_10.value).__module__}.{type(var_10.value).__qualname__}' == 'typesystem.base.Position'
    assert var_10.error is None
    module_0.ParseError(text=var_0, code=var_8, messages=var_8)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_1.__repr__()
    assert var_5 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = None
    var_8 = module_0.ValidationResult(value=var_4, error=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is True
    assert var_8.error is None
    var_9 = 'h[y'
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = module_0.BaseError(text=var_9, key=var_6, position=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_11.__hash__()
    assert var_12 == -5688242508314411925
    var_13 = var_11.__eq__(var_0)
    assert var_13 is False
    var_14 = var_13.__repr__()
    assert var_14 == 'False'
    var_15 = var_11.__str__()
    assert var_15 == "{False: 'h[y'}"
    var_16 = var_11.__str__()
    assert var_16 == "{False: 'h[y'}"
    var_17 = var_11.__repr__()
    assert var_17 == "BaseError([Message(text='h[y', code='custom', index=[False])])"
    var_18 = None
    module_0.ParseError(text=var_0, code=var_18, position=var_13, messages=var_14)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = 'ReJCV'
    var_2 = module_0.Message(text=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'ReJCV'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='ReJCV', code='custom')"
    var_5 = module_0.ValidationError(text=var_1, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = module_0.Message(text=var_6, position=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_7.text).__module__}.{type(var_7.text).__qualname__}' == 'builtins.list'
    assert len(var_7.text) == 1
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    module_0.ValidationError()

def test_case_26():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, key=var_0, index=var_0, position=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_5.__hash__()
    assert var_8 == 7523839687821003752
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_1.__eq__(var_0)
    assert var_10 is False
    var_11 = module_0.BaseError(text=var_4, key=var_6, position=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_11.get(var_0, var_0)
    var_13 = var_2.__repr__()
    assert var_13 == "Message(text=None, code='custom')"
    var_14 = var_11.__repr__()
    assert var_14 == "BaseError([Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom', index=[False])])"
    var_15 = module_0.ParseError(text=var_10, position=var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_4, index=var_0, position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Position(line_no=None, column_no=None, char_index=None)'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_5.__eq__(var_2)
    assert var_6 is False
    var_7 = var_5.__repr__()
    assert var_7 == "Message(text='Position(line_no=None, column_no=None, char_index=None)', code='custom', position=Position(line_no=None, column_no=None, char_index=None))"
    var_8 = var_1.__eq__(var_0)
    assert var_8 is False
    var_9 = var_5.__hash__()
    assert var_9 == 7523839687821003752
    var_10 = None
    var_11 = var_2.__hash__()
    assert var_11 == 7523839687821003752
    var_12 = module_0.ValidationResult(value=var_8, error=var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is False
    assert var_12.error is None
    var_13 = var_5.__eq__(var_0)
    assert var_13 is False
    var_14 = var_12.__repr__()
    assert var_14 == 'ValidationResult(value=False)'
    module_0.ParseError(code=var_0, key=var_0)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = module_0.BaseError(text=var_2, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__hash__()
    assert var_4 == -6876752870222198464
    var_5 = var_3.__str__()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_3.__repr__()
    assert var_6 == "BaseError(text=ValidationResult(value=None), code='custom')"
    module_0.ParseError(code=var_0, messages=var_0)

def test_case_29():
    var_0 = '4d3BEs&jy;'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = False
    var_4 = module_0.Position(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is False
    assert var_4.char_index is None
    var_5 = var_2.__eq__(var_1)
    assert var_5 is False

def test_case_30():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_3, index=var_4, end_position=var_0)

def test_case_31():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = 'MNO/big"glIpF{('
    var_3 = module_0.Message(text=var_2, key=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'MNO/big"glIpF{('
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = var_3.__repr__()
    assert var_6 == 'Message(text=\'MNO/big"glIpF{(\', code=\'custom\')'
    var_7 = module_0.ValidationError(text=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8) == 1
    var_9 = [var_5, var_0]
    var_10 = (var_1, var_8, var_9)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_10, error=var_8)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = 'H}W0FvdggF{T,LXy9.&'
    var_3 = None
    var_4 = module_0.Message(text=var_2, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'H}W0FvdggF{T,LXy9.&'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    module_0.ValidationError(key=var_2, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = []
    module_0.ValidationError(code=var_0, key=var_0, messages=var_1)

def test_case_34():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_2.__eq__(var_0)
    assert var_5 is False
    var_6 = 'h[y'
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.BaseError(text=var_6, key=var_5, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__hash__()
    assert var_9 == -5688242508314411925
    var_10 = var_8.__eq__(var_0)
    assert var_10 is False
    var_11 = var_10.__repr__()
    assert var_11 == 'False'
    var_12 = var_8.__str__()
    assert var_12 == "{False: 'h[y'}"
    var_13 = var_8.__str__()
    assert var_13 == "{False: 'h[y'}"
    var_14 = var_8.__repr__()
    assert var_14 == "BaseError([Message(text='h[y', code='custom', index=[False])])"

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_2.text).__module__}.{type(var_2.text).__qualname__}' == 'typesystem.base.Position'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__eq__(var_2)
    assert var_6 is True
    var_7 = var_5.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_1.__repr__()
    assert var_8 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_9 = var_1.__repr__()
    assert var_9 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_10 = module_0.ParseError(text=var_1, position=var_1)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.messages(add_prefix=var_4)
    var_12 = None
    var_13 = var_2.__hash__()
    assert var_13 == 7523839687821003752
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    module_0.ValidationError(position=var_12)

def test_case_36():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_4, index=var_0, position=var_4, start_position=var_1)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=None, char_index=None)'
    var_5 = module_0.Message(text=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = module_0.ValidationResult(error=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert f'{type(var_9.error).__module__}.{type(var_9.error).__qualname__}' == 'typesystem.base.ValidationResult'
    var_10 = module_0.BaseError(text=var_9, code=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_10.__hash__()
    assert var_11 == -6876752870222198464
    var_12 = var_10.__str__()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert f'{type(var_12.error).__module__}.{type(var_12.error).__qualname__}' == 'typesystem.base.ValidationResult'
    var_13 = var_5.__repr__()
    assert var_13 == "Message(text=None, code='custom')"
    var_14 = var_12.__str__()
    assert var_14 == 'ValidationResult(error=ValidationResult(value=None))'
    var_12.items()

def test_case_38():
    var_0 = 'invalid_json'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = 'field1'
    var_3 = module_0.Message(text=var_0, code=var_0, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'invalid_json'
    assert var_3.code == 'invalid_json'
    assert var_3.index == ['field1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Missing fieC"]ld'
    var_5 = 'missing_field'
    var_6 = 'field2'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Missing fieC"]ld'
    assert var_7.code == 'missing_field'
    assert var_7.index == ['field2']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_3, var_7]
    var_9 = module_0.ParseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 2
    var_10 = 'Invalid value'
    var_11 = 'invalid'
    var_12 = 'subfield'
    var_13 = [var_2, var_12]
    var_14 = module_0.Message(text=var_10, code=var_11, index=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'Invalid value'
    assert var_14.code == 'invalid'
    assert var_14.index == ['field1', 'subfield']
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = 'missing'
    var_16 = [var_6]
    var_17 = module_0.Message(text=var_4, code=var_15, index=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'Missing fieC"]ld'
    assert var_17.code == 'missing'
    assert var_17.index == ['field2']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = [var_14, var_17]
    var_19 = module_0.ParseError(messages=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_19) == 2
    var_20 = 'All tests passed!'
    var_21 = print(var_20)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = str(var_2)
    assert var_3 == 'Invalid input'
    var_4 = 'field1'
    var_5 = [var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Invalid input'
    assert var_6.code == 'invalid'
    assert var_6.index == ['field1']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Missing field'
    var_8 = 'missing'
    var_9 = 'field2'
    var_10 = [var_9]
    var_11 = module_0.Message(text=var_7, code=var_8, index=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'Missing field'
    assert var_11.code == 'missing'
    assert var_11.index == ['field2']
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = [var_6, var_11]
    var_13 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 2
    var_14 = str(var_13)
    var_15 = 'subfield'
    var_16 = [var_4, var_15]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'Invalid input'
    assert var_17.code == 'invalid'
    assert var_17.index == ['field1', 'subfield']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = [var_9]
    var_19 = module_0.Message(text=var_7, code=var_8, index=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text == 'Missing field'
    assert var_19.code == 'missing'
    assert var_19.index == ['field2']
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = [var_17, var_19]
    var_21 = module_0.ValidationError(messages=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_21) == 2
    var_22 = str(var_21)
    var_23 = []
    module_0.ValidationError(messages=var_23)