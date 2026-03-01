# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 1740
    var_1 = None
    var_2 = None
    var_3 = module_0.Position(var_2, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no == 1740
    assert var_3.char_index == 1740
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1740
    assert var_5.column_no == 1740
    assert var_5.char_index == 1740

def test_case_1():
    var_0 = "\r;/'gTr"
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, key=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1

def test_case_2():
    var_0 = ''
    var_1 = module_0.Message(text=var_0, code=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ''
    assert var_1.code == ''
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position == ''

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_2.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_5) == 1
    var_6 = var_2.__len__()
    assert var_6 == 1
    var_7 = var_5.__str__()
    assert var_7 == 'ValuesView(BaseError([Message(text="`fP1;o\\t0|S`\'\\tIv [", code=\'custom\', index=["`fP1;o\\t0|S`\'\\tIv ["])]))'
    var_8 = []
    var_5.get(var_8)

def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = var_2.__eq__(var_0)
    assert var_5 is False
    var_6 = var_2.__eq__(var_0)
    assert var_6 is False

@pytest.mark.xfail(strict=True)
def test_case_5():
    module_0.ParseError()

def test_case_6():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_7():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_8():
    var_0 = -105
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == -105
    assert var_1.column_no == -105
    assert var_1.char_index == -105
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_9():
    var_0 = None
    var_1 = 'sfLj(2iz|?7r'
    var_2 = module_0.ValidationError(text=var_1, code=var_1, position=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert var_3 == 'sfLj(2iz|?7r'
    var_4 = var_2.__hash__()
    assert var_4 == 1473608307143952153

def test_case_10():
    var_0 = None
    var_1 = 'sfLj(2iz|?17r'
    var_2 = module_0.ValidationError(text=var_1, code=var_1, position=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert var_3 == 'sfLj(2iz|?17r'
    var_4 = var_2.__contains__(var_3)
    assert var_4 is False

def test_case_11():
    var_0 = []
    var_1 = (528.3+1161.8145j)
    var_2 = module_0.Message(text=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == []
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == (528.3+1161.8145j)
    assert var_2.end_position == (528.3+1161.8145j)
    var_3 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == []
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == []
    assert var_3.end_position == []
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__bool__()
    assert var_5 is True
    var_6 = var_2.__str__()
    assert var_6 == "Message(text=[], code='custom', position=(528.3+1161.8145j))"
    var_7 = var_2.__hash__()
    assert var_7 == 1046332209453632671

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = '/>Ao3{r$&Tv'
    var_1 = module_0.ValidationError(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()
    var_2.__len__()

def test_case_13():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_3, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = module_0.ValidationError(text=var_0, position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_4.__eq__(var_1)
    assert var_8 is False
    var_9 = var_6.values()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_9) == 1
    var_10 = var_2.__len__()
    assert var_10 == 1
    var_11 = var_4.__eq__(var_1)
    assert var_11 is False
    var_12 = var_6.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_12.__repr__()
    assert var_13 == 'KeysView(ValidationError(text="`fP1;o\\t0|S`\'\\tIv [", code=\'custom\'))'

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = 'Y%@Uz'
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = -952
    var_3 = var_1.messages()
    var_4 = var_1.messages(add_prefix=var_0)
    var_5 = False
    var_6 = var_1.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6) == 1
    var_7 = var_6.__eq__(var_0)
    var_8 = module_0.Position(var_2, var_5, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == -952
    assert var_8.column_no is False
    assert var_8.char_index == 'Y%@Uz'
    var_6.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'I?>aHV\n'
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'I?>aHV\n'
    assert var_1.code == 'custom'
    assert var_1.index == 'I?>aHV\n'
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    module_0.ParseError(text=var_2)

def test_case_16():
    var_0 = None
    var_1 = 'sfL(2iz|?7r'
    var_2 = module_0.ValidationError(text=var_1, code=var_1, position=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__hash__()
    assert var_4 == 5499385311186603637

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = None
    var_2 = []
    module_0.ValidationError(code=var_1, key=var_1, messages=var_2)

def test_case_18():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.ValidationResult(value=var_1, error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_2.value).__module__}.{type(var_2.value).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.error is None

def test_case_19():
    var_0 = None
    var_1 = 'I?>aHV\n'
    var_2 = module_0.Message(text=var_1, index=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'I?>aHV\n'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = False
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, position=var_4, messages=var_1)

def test_case_20():
    var_0 = '\x0crf%q?Y'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = var_2.__bool__()
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = []
    var_1 = "`fP1;o\t0|S`'\tIv ["
    var_2 = None
    var_3 = module_0.BaseError(text=var_1, code=var_2, key=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_2, position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = None
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = var_3.__str__()
    assert var_7 == '{"`fP1;o\\t0|S`\'\\tIv [": "`fP1;o\\t0|S`\'\\tIv ["}'
    var_8 = var_3.__contains__(var_2)
    assert var_8 is False
    var_9 = var_4.__eq__(var_5)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is False
    assert var_10.error is None
    var_11 = var_7.__repr__()
    assert var_11 == '\'{"`fP1;o\\\\t0|S`\\\'\\\\tIv [": "`fP1;o\\\\t0|S`\\\'\\\\tIv ["}\''
    var_12 = None
    var_13 = var_8.__eq__(var_2)
    module_0.ParseError(code=var_2, position=var_12)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = '/>Ao3{r$&Tv'
    var_1 = module_0.ValidationError(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '/>Ao3{r$&Tv'
    var_3 = var_2.__len__()
    assert var_3 == 11
    var_3.__len__()

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.__len__()
    assert var_4 == 1
    var_5 = var_2.__iter__()
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    module_0.ValidationError(code=var_0, key=var_5, position=var_1, messages=var_3)

def test_case_24():
    var_0 = 'Error message'
    var_1 = 'custom'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = 42
    var_5 = module_0.ValidationResult(value=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == 42
    assert var_5.error is None
    var_6 = 'Error 2'
    var_7 = 'error2'
    var_8 = 'field2'
    var_9 = module_0.Message(text=var_6, code=var_7, key=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'error2'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_9, var_9]
    var_11 = module_0.ValidationError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11) == 1
    var_12 = module_0.ValidationResult(error=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert f'{type(var_12.error).__module__}.{type(var_12.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12.error) == 1
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = iter(var_13)
    var_15 = next(var_14)
    var_16 = next(var_14)
    with pytest.raises(StopIteration):
        var_17 = next(var_14)

def test_case_25():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = var_2.__len__()
    assert var_4 == 1
    var_5 = module_0.Message(text=var_3, position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__hash__()
    assert var_6 == 1046332209453632671
    var_7 = None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_8.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = var_5.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = None
    var_12 = module_0.ValidationResult(value=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value == 1046332209453632671
    assert var_12.error is None
    var_13 = module_0.ParseError(text=var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.__str__()
    assert var_14 == "Message(text=None, code='custom')"
    var_15 = False
    var_16 = module_0.Position(var_11, var_7, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no is None
    assert var_16.column_no is None
    assert var_16.char_index is False
    var_17 = var_14.__eq__(var_1)
    var_18 = var_5.__hash__()
    assert var_18 == 1046332209453632671
    var_19 = var_16.__repr__()
    assert var_19 == 'Position(line_no=None, column_no=None, char_index=False)'

def test_case_26():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.__iter__()
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_6 = var_5.__repr__()
    assert var_6 == '1'
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_3.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_9.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_3.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = None
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = module_0.ParseError(text=var_6, messages=var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_2.__eq__(var_2)
    assert var_15 is False
    var_16 = False
    var_17 = False
    var_18 = module_0.Position(var_17, var_16, var_4)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Position'
    assert var_18.line_no is False
    assert var_18.column_no is False
    assert f'{type(var_18.char_index).__module__}.{type(var_18.char_index).__qualname__}' == 'builtins.dict_keyiterator'
    var_19 = var_4.__eq__(var_0)
    var_20 = var_14.keys()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_20) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, code=var_4, key=var_5, index=var_1, position=var_20, end_position=var_5)

def test_case_27():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_0, key=var_0, messages=var_0)

def test_case_28():
    var_0 = []
    var_1 = "`fP1;o\t0|S`'\tIv ["
    var_2 = None
    var_3 = module_0.BaseError(text=var_1, code=var_2, key=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.get(var_2, var_2)
    var_5 = var_4.__hash__()
    assert var_5 == 8640208727960
    var_6 = module_0.Message(text=var_2, position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__len__()
    assert var_7 == 1
    var_8 = None
    var_9 = var_6.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_3.__eq__(var_10)
    assert var_11 is False
    var_12 = var_3.__str__()
    assert var_12 == '{"`fP1;o\\t0|S`\'\\tIv [": "`fP1;o\\t0|S`\'\\tIv ["}'
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_6.__eq__(var_8)
    assert var_14 is False
    var_15 = module_0.ValidationResult(value=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_15.value).__module__}.{type(var_15.value).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.error is None
    var_16 = var_3.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_16.__repr__()
    assert var_17 == 'KeysView(BaseError([Message(text="`fP1;o\\t0|S`\'\\tIv [", code=\'custom\', index=["`fP1;o\\t0|S`\'\\tIv ["])]))'
    var_18 = module_0.ValidationResult()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.value is None
    assert var_18.error is None
    var_19 = var_11.__eq__(var_8)
    var_20 = module_0.Position(var_2, var_7, var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no is None
    assert var_20.column_no == 1
    assert var_20.char_index is None
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_19, position=var_2, messages=var_16)

def test_case_29():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = []
    var_4 = [var_0, var_0, var_0]
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_3, index=var_4, start_position=var_1)

def test_case_30():
    var_0 = []
    var_1 = "`fP1;o\t0|S`'\tIv ["
    var_2 = None
    var_3 = module_0.BaseError(text=var_1, code=var_2, key=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_2, position=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_3.__len__()
    assert var_5 == 1
    var_6 = None
    var_7 = var_4.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_3.__eq__(var_6)
    assert var_9 is False
    var_10 = True
    var_11 = (var_10,)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_11, error=var_9)

def test_case_31():
    var_0 = None
    var_1 = (528.3+1161.8145j)
    var_2 = module_0.Message(text=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == (528.3+1161.8145j)
    assert var_2.end_position == (528.3+1161.8145j)
    var_3 = "`fP1;o\t0|S`'\tIv ["
    var_4 = None
    var_5 = module_0.BaseError(text=var_3, code=var_4, key=var_3, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text=None, code='custom', position=(528.3+1161.8145j))"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0)

def test_case_32():
    var_0 = 'e!<<}=vw-b)7D/'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'e!<<}=vw-b)7D/'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

def test_case_33():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = repr(var_1)
    assert var_2 == "ValidationResult(value='test_value')"
    var_3 = 'Error message'
    var_4 = 'custom'
    var_5 = module_0.ValidationError(text=var_3, code=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = repr(var_6)
    assert var_7 == "ValidationResult(error=ValidationError(text='Error message', code='custom'))"
    var_8 = 'First error'
    var_9 = 'max_length'
    var_10 = module_0.Message(text=var_8, code=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'First error'
    assert var_10.code == 'max_length'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = 'Second error'
    var_12 = 'min_length'
    var_13 = module_0.Message(text=var_11, code=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'Second error'
    assert var_13.code == 'min_length'
    assert var_13.index == []
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = [var_10, var_13]
    var_15 = module_0.ValidationError(messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = module_0.ValidationResult(error=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert f'{type(var_16.error).__module__}.{type(var_16.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16.error) == 1
    var_17 = repr(var_16)
    var_18 = repr(var_16)
    var_19 = repr(var_16)
    var_20 = 'Field error'
    var_21 = 'username'
    var_22 = module_0.ValidationError(text=var_20, key=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_22) == 1
    var_23 = module_0.ValidationResult(error=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_23.value is None
    assert f'{type(var_23.error).__module__}.{type(var_23.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_23.error) == 1
    var_24 = repr(var_23)
    assert var_24 == "ValidationResult(error=ValidationError([Message(text='Field error', code='custom', index=['username'])]))"
    var_25 = None
    var_26 = module_0.ValidationResult(value=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_26.value is None
    assert var_26.error is None
    var_27 = repr(var_26)
    assert var_27 == 'ValidationResult(value=None)'

def test_case_34():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = repr(var_1)
    assert var_2 == "ValidationResult(value='test_value')"
    var_3 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error == "ValidationResult(value='test_value')"
    var_4 = 'Field required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_4, code=var_5, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Field required'
    assert var_7.code == 'required'
    assert var_7.index == ['username']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = 'Too short'
    var_9 = 'min_length'
    var_10 = 'password'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'Too short'
    assert var_11.code == 'min_length'
    assert var_11.index == ['password']
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = [var_7, var_11]
    var_13 = module_0.ValidationError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 2
    var_14 = module_0.ValidationResult(error=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert f'{type(var_14.error).__module__}.{type(var_14.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14.error) == 2
    var_15 = repr(var_14)
    var_16 = ''
    var_17 = module_0.ValidationResult(value=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value == ''
    assert var_17.error is None
    var_18 = repr(var_17)
    assert var_18 == "ValidationResult(value='')"
    var_19 = 42
    var_20 = module_0.ValidationResult(value=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_20.value == 42
    assert var_20.error is None
    var_21 = repr(var_20)
    assert var_21 == 'ValidationResult(value=42)'
    var_22 = 'Invalid email'
    var_23 = 'invalid_email'
    var_24 = 'users'
    var_25 = 'email'
    var_26 = [var_24, var_18, var_25]
    var_27 = 1
    var_28 = 5
    var_29 = 4
    var_30 = module_0.Position(var_27, var_28, var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.Position'
    assert var_30.line_no == 1
    assert var_30.column_no == 5
    assert var_30.char_index == 4
    var_31 = module_0.Message(text=var_22, code=var_23, index=var_26, position=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.base.Message'
    assert var_31.text == 'Invalid email'
    assert var_31.code == 'invalid_email'
    assert var_31.index == ['users', "ValidationResult(value='')", 'email']
    assert f'{type(var_31.start_position).__module__}.{type(var_31.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_31.end_position).__module__}.{type(var_31.end_position).__qualname__}' == 'typesystem.base.Position'
    var_32 = [var_31]
    var_33 = module_0.ValidationError(messages=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_33) == 1
    var_34 = module_0.ValidationResult(error=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_34.value is None
    assert f'{type(var_34.error).__module__}.{type(var_34.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_34.error) == 1
    var_35 = repr(var_34)
    var_36 = repr(var_34)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = []
    var_1 = None
    var_2 = (528.3+1161.8145j)
    var_3 = module_0.Message(text=var_1, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == (528.3+1161.8145j)
    assert var_3.end_position == (528.3+1161.8145j)
    var_4 = "`fP1;o\t0|S`'\tIv ["
    var_5 = None
    var_6 = module_0.BaseError(text=var_4, code=var_5, key=var_4, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text=None, code='custom', position=(528.3+1161.8145j))"
    var_8 = None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = var_6.__iter__()
    var_11 = var_6.__eq__(var_7)
    assert var_11 is False
    var_12 = var_6.__str__()
    assert var_12 == '{"`fP1;o\\t0|S`\'\\tIv [": "`fP1;o\\t0|S`\'\\tIv ["}'
    var_13 = var_6.__contains__(var_5)
    assert var_13 is False
    var_14 = var_7.__eq__(var_8)
    var_15 = module_0.ValidationResult(value=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is False
    assert var_15.error is None
    var_16 = module_0.Message(text=var_1, start_position=var_11)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position is False
    assert var_16.end_position is None
    var_17 = var_16.__repr__()
    assert var_17 == "Message(text=None, code='custom', start_position=False, end_position=None)"
    var_18 = module_0.ValidationResult()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.value is None
    assert var_18.error is None
    var_16.__len__()

def test_case_36():
    var_0 = []
    var_1 = None
    var_2 = (528.3+1161.8145j)
    var_3 = module_0.Message(text=var_1, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == (528.3+1161.8145j)
    assert var_3.end_position == (528.3+1161.8145j)
    var_4 = 'z>Xuv\t3\tUO#9tC'
    var_5 = None
    var_6 = module_0.BaseError(text=var_4, code=var_5, key=var_4, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text=None, code='custom', position=(528.3+1161.8145j))"
    var_8 = None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = var_6.__iter__()
    var_11 = var_6.__eq__(var_7)
    assert var_11 is False
    var_12 = var_6.__str__()
    assert var_12 == "{'z>Xuv\\t3\\tUO#9tC': 'z>Xuv\\t3\\tUO#9tC'}"
    var_13 = var_7.__eq__(var_8)
    var_14 = None
    var_15 = module_0.ValidationResult(value=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_15.value).__module__}.{type(var_15.value).__qualname__}' == 'builtins.NotImplementedType'
    assert var_15.error is None
    var_16 = var_6.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_16.__repr__()
    assert var_17 == "KeysView(BaseError([Message(text='z>Xuv\\t3\\tUO#9tC', code='custom', index=['z>Xuv\\t3\\tUO#9tC'])]))"
    var_18 = module_0.ValidationResult()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.value is None
    assert var_18.error is None
    var_19 = var_11.__hash__()
    assert var_19 == 0
    var_20 = var_16.__len__()
    assert var_20 == 1
    var_21 = var_15.__repr__()
    assert var_21 == 'ValidationResult(value=NotImplemented)'
    var_22 = module_0.Position(var_5, var_3, var_14)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert var_22.line_no is None
    assert f'{type(var_22.column_no).__module__}.{type(var_22.column_no).__qualname__}' == 'typesystem.base.Message'
    assert var_22.char_index is None
    var_23 = '\t'
    var_24 = None
    var_25 = var_22.__eq__(var_22)
    assert var_25 is True
    var_26 = var_11.__hash__()
    assert var_26 == 0
    var_27 = var_16.__len__()
    assert var_27 == 1
    var_28 = var_7.__eq__(var_14)
    with pytest.raises(AssertionError):
        module_0.Message(text=var_23, key=var_24, position=var_11, start_position=var_16, end_position=var_11)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = "`fP1;o\t0|S`'\tIv ["
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()
    var_4 = var_2.__str__()
    assert var_4 == '{"`fP1;o\\t0|S`\'\\tIv [": "`fP1;o\\t0|S`\'\\tIv ["}'
    var_5 = None
    var_6 = var_2.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6) == 1
    var_7 = var_6.__repr__()
    assert var_7 == 'KeysView(BaseError([Message(text="`fP1;o\\t0|S`\'\\tIv [", code=\'custom\', index=["`fP1;o\\t0|S`\'\\tIv ["])]))'
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_6.__len__()
    assert var_9 == 1
    var_10 = var_6.__len__()
    assert var_10 == 1
    var_11 = module_0.Position(var_1, var_2, var_5)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert f'{type(var_11.column_no).__module__}.{type(var_11.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11.column_no) == 1
    assert var_11.char_index is None
    var_12 = var_11.__eq__(var_11)
    assert var_12 is False
    var_13 = var_6.__len__()
    assert var_13 == 1
    var_13.items()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = str(var_2)
    assert var_3 == 'Invalid value'
    var_4 = 'Field is required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.BaseError(text=var_4, code=var_5, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = str(var_7)
    assert var_8 == "{'username': 'Field is required'}"
    var_9 = 'Invalid email'
    var_10 = 'email'
    var_11 = [var_10]
    var_12 = module_0.Message(text=var_9, code=var_1, index=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'Invalid email'
    assert var_12.code == 'invalid'
    assert var_12.index == ['email']
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = 'Too short'
    var_14 = 'min_length'
    var_15 = 'password'
    var_16 = [var_15]
    var_17 = module_0.Message(text=var_13, code=var_14, index=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'Too short'
    assert var_17.code == 'min_length'
    assert var_17.index == ['password']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = [var_12, var_17]
    var_19 = module_0.BaseError(messages=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_19) == 2
    var_20 = str(var_19)
    assert var_20 == "{'email': 'Invalid email', 'password': 'Too short'}"
    var_21 = 'users'
    var_22 = 0
    var_23 = [var_21, var_22, var_10]
    var_24 = module_0.Message(text=var_0, code=var_1, index=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Message'
    assert var_24.text == 'Invalid value'
    assert var_24.code == 'invalid'
    assert var_24.index == ['users', 0, 'email']
    assert var_24.start_position is None
    assert var_24.end_position is None
    var_25 = 'Required field'
    var_26 = 1
    var_27 = 'name'
    var_28 = [var_21, var_26, var_27]
    var_29 = module_0.Message(text=var_25, code=var_5, index=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Message'
    assert var_29.text == 'Required field'
    assert var_29.code == 'required'
    assert var_29.index == ['users', 1, 'name']
    assert var_29.start_position is None
    assert var_29.end_position is None
    module_0.BaseError(messages=var_28)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = []
    var_1 = None
    var_2 = (528.3+1161.8145j)
    var_3 = module_0.Message(text=var_1, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == (528.3+1161.8145j)
    assert var_3.end_position == (528.3+1161.8145j)
    var_4 = 'z>Xuv\t3\tUO#9tC'
    var_5 = None
    var_6 = module_0.BaseError(text=var_4, code=var_5, key=var_4, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text=None, code='custom', position=(528.3+1161.8145j))"
    var_8 = None
    var_9 = var_3.__eq__(var_0)
    assert var_9 is False
    var_10 = var_6.__iter__()
    var_11 = var_6.__eq__(var_7)
    assert var_11 is False
    var_12 = var_6.__str__()
    assert var_12 == "{'z>Xuv\\t3\\tUO#9tC': 'z>Xuv\\t3\\tUO#9tC'}"
    var_13 = var_7.__eq__(var_8)
    var_14 = None
    var_15 = module_0.ValidationResult(value=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_15.value).__module__}.{type(var_15.value).__qualname__}' == 'builtins.NotImplementedType'
    assert var_15.error is None
    var_16 = var_6.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_16.__repr__()
    assert var_17 == "KeysView(BaseError([Message(text='z>Xuv\\t3\\tUO#9tC', code='custom', index=['z>Xuv\\t3\\tUO#9tC'])]))"
    var_18 = module_0.ValidationResult()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.value is None
    assert var_18.error is None
    var_19 = var_11.__hash__()
    assert var_19 == 0
    var_20 = var_16.__len__()
    assert var_20 == 1
    var_21 = var_15.__repr__()
    assert var_21 == 'ValidationResult(value=NotImplemented)'
    var_22 = module_0.Position(var_5, var_3, var_14)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert var_22.line_no is None
    assert f'{type(var_22.column_no).__module__}.{type(var_22.column_no).__qualname__}' == 'typesystem.base.Message'
    assert var_22.char_index is None
    var_23 = -5654
    var_24 = module_0.Position(var_23, var_1, var_9)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Position'
    assert var_24.line_no == -5654
    assert var_24.column_no is None
    assert var_24.char_index is False
    var_25 = var_22.__eq__(var_24)
    assert var_25 is False
    var_26 = var_11.__hash__()
    assert var_26 == 0
    var_27 = var_6.__len__()
    assert var_27 == 1
    var_28 = var_20.__eq__(var_14)
    var_16.items()

def test_case_40():
    var_0 = None
    var_1 = 'sfL(2iz|?7r'
    var_2 = module_0.ValidationError(text=var_1, code=var_1, position=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_3.__repr__()
    assert var_4 == 'True'
    var_5 = var_2.__eq__(var_0)
    assert var_5 is False
    var_6 = var_2.__hash__()
    assert var_6 == 5499385311186603637