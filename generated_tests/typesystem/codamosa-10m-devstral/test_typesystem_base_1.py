# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = '\n$,8tp"7r'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.__str__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__len__()
    assert var_6 == 1
    var_7 = var_4.messages(add_prefix=var_6)
    var_8 = var_4.__iter__()
    var_9 = module_0.BaseError(text=var_7, position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_5.__eq__(var_3)
    assert var_10 is True
    var_11 = module_0.Message(text=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == "Message(text=None, code='custom')"
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_6.__repr__()
    assert var_12 == '1'
    var_13 = var_6.__repr__()
    assert var_13 == '1'
    var_14 = var_8.__repr__()

def test_case_2():
    var_0 = 'a&$$=L?~>'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'a&$$=L?~>'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = "4`'KUM;"
    var_1 = module_0.BaseError(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()
    var_2.items()

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ValidationResult()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_5():
    module_0.ParseError()

def test_case_6():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_7():
    var_0 = False
    var_1 = None
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is None
    assert var_2.char_index is None

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == -4554019133227855076
    var_3 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.Message'
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    module_0.ParseError(code=var_0, key=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.ParseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__repr__()
    assert var_4 == "ParseError(text='ValidationResult(value=None)', code='custom')"
    var_5 = var_3.__iter__()
    module_0.ParseError(position=var_0)

def test_case_10():
    var_0 = '\n$,8tp"7r'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == 'ParseError(text=\'\\n$,8tp"7r\', code=\'\\n$,8tp"7r\')'
    var_5 = var_1.__contains__(var_2)
    assert var_5 is False
    var_6 = var_5.__eq__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.ParseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_1.__bool__()
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == "ParseError(text='ValidationResult(value=None)', code='custom')"
    module_0.ParseError()

def test_case_13():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.Message(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = -1352
    var_5 = module_0.Position(var_4, var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == -1352
    assert var_5.column_no == -1352
    assert var_5.char_index is None
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.BaseError(text=var_5, code=var_0, position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_2.__repr__()
    assert var_8 == 'False'
    var_9 = module_0.ParseError(text=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__repr__()
    assert var_10 == 'ParseError(text="Message(text=None, code=\'custom\')", code=\'custom\')'
    var_11 = var_6.messages()
    var_12 = var_6.__iter__()
    var_13 = var_1.__hash__()
    assert var_13 == -4554019133227855076
    var_14 = var_1.__hash__()
    assert var_14 == -4554019133227855076
    var_15 = var_12.__iter__()
    var_16 = var_3.__repr__()
    assert var_16 == 'ValidationResult(value=None)'
    module_0.ParseError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.Message'
    assert var_3.error is None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = var_3.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = module_0.ParseError(text=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_6.__len__()
    assert var_9 == 1
    var_9.keys()

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = {}
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == {}
    assert var_1.error is None
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = []
    module_0.ParseError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = -164
    var_6 = module_0.Position(var_2, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no == -164
    assert var_6.char_index == -164
    var_7 = var_4.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_1.__str__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_3.__len__()
    assert var_9 == 33
    var_10 = module_0.ParseError(text=var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == "ParseError(text='ValidationResult(value=None)', code='custom')"
    var_12 = var_10.messages(add_prefix=var_7)
    var_13 = var_10.__iter__()
    var_14 = module_0.BaseError(text=var_12, position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_10.__eq__(var_13)
    assert var_15 is False
    var_16 = var_10.__contains__(var_13)
    assert var_16 is False
    var_17 = module_0.ParseError(messages=var_12)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_17) == 1
    var_18 = var_6.__eq__(var_6)
    assert var_18 is True
    var_19 = var_17.values()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_19) == 1
    var_20 = var_10.__hash__()
    assert var_20 == -6889161311057812408
    var_21 = var_17.__contains__(var_2)
    assert var_21 is False
    var_22 = var_19.__repr__()
    assert var_22 == "ValuesView(ParseError([Message(text='ValidationResult(value=None)', code='custom', index=['ValidationResult(value=None)'])]))"
    var_23 = var_21.__repr__()
    assert var_23 == 'False'
    var_24 = var_14.__contains__(var_22)
    assert var_24 is False
    module_0.ParseError(text=var_19, key=var_21, position=var_13, messages=var_12)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Message(text=var_0)
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
    var_6 = var_5.__repr__()
    assert var_6 == 'ValidationResult(value=None)'
    var_7 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_7.__eq__(var_0)
    assert var_8 is False
    var_9 = var_8.__repr__()
    assert var_9 == 'False'
    var_10 = module_0.Message(text=var_0, code=var_3, index=var_0, position=var_8, end_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == "Message(text=None, code='custom')"
    assert var_10.index == []
    assert var_10.start_position is False
    assert var_10.end_position is False
    var_11 = var_7.__iter__()
    var_12 = var_0.__repr__()
    assert var_12 == 'None'
    module_0.ParseError(code=var_11, key=var_3, messages=var_11)

def test_case_20():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = module_0.ValidationResult(error=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.Message'
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = 't&ln8='
    var_8 = [var_2, var_6]
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_7, index=var_8, position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    module_0.ParseError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = "~c'>"
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.ParseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ParseError(text=ValidationResult(value=None), code='custom')"
    var_4 = {var_3, var_2, var_1}
    var_5 = var_2.__iter__()
    var_2.__contains__(var_4)

def test_case_24():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is True
    assert var_4.column_no == "Message(text=None, code='custom')"
    assert var_4.char_index == "Message(text=None, code='custom')"
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_3.__len__()
    assert var_8 == 33
    var_9 = var_1.__hash__()
    assert var_9 == -4554019133227855076
    var_10 = var_4.__eq__(var_4)
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Message(text=var_0)
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
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_1.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_5.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    module_0.ParseError(position=var_0, messages=var_3)

def test_case_26():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Message(text=var_0)
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
    var_6 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_6.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__eq__(var_0)
    assert var_10 is False
    var_11 = var_10.__repr__()
    assert var_11 == 'False'
    var_12 = 'R0~'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_12, key=var_10, position=var_10, start_position=var_10, end_position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_6 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = var_6.__iter__()
    var_10 = var_0.__repr__()
    assert var_10 == 'None'
    module_0.ParseError(code=var_9, key=var_3, messages=var_9)

def test_case_28():
    var_0 = '\n$,8tp"7r'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == 'ParseError(text=\'\\n$,8tp"7r\', code=\'\\n$,8tp"7r\')'

def test_case_29():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Message(text=var_3, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == "Message(text=None, code='custom')"
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_1, error=var_2)

def test_case_30():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.Message(text=var_0)
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
    var_6 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = var_8.__contains__(var_0)
    assert var_10 is False
    var_11 = var_10.__repr__()
    assert var_11 == 'False'
    var_12 = -1814
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0, key=var_10, position=var_9, messages=var_12)

def test_case_31():
    var_0 = '\n,8tp"7r'
    var_1 = module_0.ParseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1

def test_case_32():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_5 = module_0.Message(text=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = module_0.ValidationResult(error=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationResult'
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_6.__repr__()
    assert var_8 == 'ValidationResult(error=ValidationResult(value=None))'
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_0, messages=var_0)

def test_case_33():
    var_0 = 'Error message'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error message'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = repr(var_1)
    assert var_2 == "Message(text='Error message', code='custom')"
    var_3 = 'username'
    var_4 = module_0.Message(text=var_0, code=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error message'
    assert var_4.code == 'Error message'
    assert var_4.index == ['username']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = repr(var_4)
    var_6 = 'users'
    var_7 = 3
    var_8 = [var_6, var_7, var_3]
    var_9 = module_0.Message(text=var_0, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error message'
    assert var_9.code == 'custom'
    assert var_9.index == ['users', 3, 'username']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = repr(var_9)
    assert var_10 == "Message(text='Error message', code='custom', index=['users', 3, 'username'])"
    var_11 = 1
    var_12 = 2
    var_13 = module_0.Position(var_11, var_12, var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 1
    assert var_13.column_no == 2
    assert var_13.char_index == 3
    var_14 = module_0.Message(text=var_0, position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'Error message'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert f'{type(var_14.start_position).__module__}.{type(var_14.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'typesystem.base.Position'
    var_15 = repr(var_14)
    assert var_15 == "Message(text='Error message', code='custom', position=Position(line_no=1, column_no=2, char_index=3))"
    var_16 = module_0.Position(var_11, var_12, var_7)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no == 1
    assert var_16.column_no == 2
    assert var_16.char_index == 3
    var_17 = 5
    var_18 = 8
    var_19 = module_0.Position(var_11, var_17, var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no == 1
    assert var_19.column_no == 5
    assert var_19.char_index == 8
    var_20 = module_0.Message(text=var_0, start_position=var_16, end_position=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'Error message'
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert f'{type(var_20.start_position).__module__}.{type(var_20.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_20.end_position).__module__}.{type(var_20.end_position).__qualname__}' == 'typesystem.base.Position'
    var_21 = repr(var_20)

def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__str__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_5.__contains__(var_0)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = module_0.ParseError(text=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__repr__()
    assert var_10 == "ParseError(text=False, code='custom')"
    var_11 = var_5.messages(add_prefix=var_7)
    var_12 = {var_9, var_10, var_9, var_9}
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_12, messages=var_7)

def test_case_35():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_6.__contains__(var_0)
    assert var_8 is False
    var_9 = var_6.__len__()
    assert var_9 == 1
    var_10 = module_0.ParseError(text=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == "ParseError(text=False, code='custom')"
    var_12 = var_6.messages(add_prefix=var_8)
    var_13 = var_6.__iter__()
    var_14 = module_0.BaseError(text=var_12, position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_9.__hash__()
    assert var_15 == 1
    var_16 = 'dQ+\x0bqv/{\\'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_13, code=var_0, key=var_5, position=var_16, end_position=var_9)

def test_case_36():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'Error 1'
    var_6 = 'code1'
    var_7 = 'key1'
    var_8 = module_0.Message(text=var_5, code=var_6, key=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error 1'
    assert var_8.code == 'code1'
    assert var_8.index == ['key1']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = 'Error 2'
    var_10 = 'code2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_9, code=var_10, key=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'Error 2'
    assert var_12.code == 'code2'
    assert var_12.index == ['key2']
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = [var_8, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 2
    var_15 = len(var_14)
    assert var_15 == 2
    var_16 = 'Nested error'
    var_17 = 'parent'
    var_18 = module_0.BaseError(text=var_16, key=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_18) == 1
    var_19 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_19) == 1
    var_20 = repr(var_3)
    var_21 = repr(var_14)
    var_22 = str(var_3)
    var_23 = str(var_14)
    var_24 = dict(var_14)
    var_25 = str(var_24)
    var_26 = var_3.messages()
    var_27 = 0
    var_28 = "/KTljR<';3T=8"
    with pytest.raises(NameError):
        var_29 = error1.messages(add_prefix=var_28)[var_27]

def test_case_37():
    var_0 = -164
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == -164
    assert var_1.column_no == -164
    assert var_1.char_index == -164
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_38():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = 'Error 1'
    var_4 = 'code1'
    var_5 = 'key1'
    var_6 = module_0.Message(text=var_3, code=var_4, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 1'
    assert var_6.code == 'code1'
    assert var_6.index == ['key1']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = 'key2'
    var_10 = module_0.Message(text=var_7, code=var_8, key=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'Error 2'
    assert var_10.code == 'code2'
    assert var_10.index == ['key2']
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = [var_6, var_10]
    var_12 = module_0.BaseError(messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 2
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'Nested error'
    var_15 = 'parent'
    var_16 = module_0.BaseError(text=var_14, key=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = repr(var_16)
    var_19 = repr(var_12)
    var_20 = str(var_19)
    var_21 = str(var_12)
    var_22 = dict(var_12)
    var_23 = str(var_22)
    var_24 = var_17.messages()
    var_25 = 0
    var_26 = 'prefix'
    with pytest.raises(NameError):
        var_27 = error1.messages(add_prefix=var_26)[var_25]

def test_case_39():
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
    var_6 = 'test_code'
    var_7 = module_0.ValidationError(text=var_5, code=var_6)
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
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = iter(var_12)
    var_14 = next(var_13)
    assert var_14 is None

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = -147
    var_6 = module_0.Position(var_2, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no == -147
    assert var_6.char_index == -147
    var_7 = var_6.__repr__()
    assert var_7 == 'Position(line_no=True, column_no=-147, char_index=-147)'
    var_8 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = module_0.ParseError(text=var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_4.__bool__()
    assert var_11 is True
    var_12 = var_10.__repr__()
    assert var_12 == "ParseError(text='Position(line_no=True, column_no=-147, char_index=-147)', code='custom')"
    var_13 = var_8.messages(add_prefix=var_7)
    var_14 = var_8.__iter__()
    var_15 = module_0.BaseError(text=var_13, position=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = var_10.__eq__(var_14)
    assert var_16 is False
    var_17 = var_14.__repr__()
    var_18 = var_15.__repr__()
    assert var_18 == 'BaseError(text=[Message(text="Message(text=None, code=\'custom\')", code=\'custom\', index=[\'Position(line_no=True, column_no=-147, char_index=-147)\'])], code=\'custom\')'
    var_19 = module_0.ParseError(text=var_14, code=var_0, key=var_0, position=var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_19) == 1
    var_20 = module_0.Position(var_0, var_14, var_14)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no is None
    assert f'{type(var_20.column_no).__module__}.{type(var_20.column_no).__qualname__}' == 'builtins.dict_keyiterator'
    assert f'{type(var_20.char_index).__module__}.{type(var_20.char_index).__qualname__}' == 'builtins.dict_keyiterator'
    var_21 = var_6.__eq__(var_20)
    assert var_21 is False
    var_22 = var_4.__iter__()
    var_23 = var_8.values()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_23) == 1
    var_14.messages()

def test_case_41():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = -147
    var_6 = module_0.Position(var_2, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no == -147
    assert var_6.char_index == -147
    var_7 = var_4.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_1.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_9.__str__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = module_0.ParseError(text=var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_9.items()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_13) == 1
    var_14 = var_12.__repr__()
    assert var_14 == "ParseError(text='ValidationResult(value=None)', code='custom')"
    var_15 = var_9.messages(add_prefix=var_8)
    var_16 = module_0.ValidationError(text=var_13, code=var_8, key=var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16) == 1
    var_17 = module_0.BaseError(text=var_15, position=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = var_12.__eq__(var_16)
    assert var_18 is False
    var_19 = var_4.__repr__()
    assert var_19 == 'ValidationResult(value=None)'
    var_20 = module_0.ParseError(messages=var_15)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_20) == 1
    var_21 = var_6.__eq__(var_6)
    assert var_21 is True
    var_22 = var_4.__iter__()
    var_23 = var_17.values()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_23) == 1
    var_24 = var_17.messages()
    var_25 = var_24.__repr__()
    assert var_25 == '[Message(text=[Message(text="Message(text=None, code=\'custom\')", code=\'custom\', index=[\'ValidationResult(value=None)\'])], code=\'custom\', position=ValidationError([Message(text=ItemsView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\')), code=\'ValidationResult(value=None)\', index=[True])]))]'
    var_26 = var_20.__contains__(var_0)
    assert var_26 is False
    var_27 = var_20.__str__()
    assert var_27 == '{\'ValidationResult(value=None)\': "Message(text=None, code=\'custom\')"}'
    var_28 = var_9.__contains__(var_23)
    assert var_28 is False
    var_29 = module_0.ParseError(text=var_3, position=var_9)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_29) == 1

def test_case_42():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = 'error_key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = module_0.Position(var_5, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == 1
    assert var_8.column_no == 2
    assert var_8.char_index == 3
    var_9 = module_0.BaseError(text=var_0, position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = 'Error 1'
    var_11 = 'code1'
    var_12 = 'key1'
    var_13 = module_0.Message(text=var_10, code=var_11, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'Error 1'
    assert var_13.code == 'code1'
    assert var_13.index == ['key1']
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = [var_13, var_13]
    var_15 = module_0.BaseError(messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = len(var_15)
    var_17 = 'Nested error'
    var_18 = 'parent'
    var_19 = 'child'
    var_20 = [var_18, var_19]
    var_21 = module_0.Message(text=var_17, index=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text == 'Nested error'
    assert var_21.code == 'custom'
    assert var_21.index == ['parent', 'child']
    assert var_21.start_position is None
    assert var_21.end_position is None
    var_22 = [var_21]
    var_23 = module_0.BaseError(messages=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_23) == 1

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0)
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
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no is True
    assert var_6.char_index is True
    var_7 = var_5.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = module_0.BaseError(text=var_3, code=var_0, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = module_0.ParseError(text=var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_5.__bool__()
    assert var_12 is True
    var_13 = var_11.__repr__()
    assert var_13 == "ParseError(text='ValidationResult(value=None)', code='custom')"
    var_14 = var_9.messages(add_prefix=var_7)
    var_15 = var_9.__iter__()
    var_16 = module_0.BaseError(text=var_14, position=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = var_1.__eq__(var_0)
    assert var_17 is False
    var_18 = var_11.__eq__(var_15)
    assert var_18 is False
    var_19 = var_15.__repr__()
    var_20 = module_0.ParseError(messages=var_14)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_20) == 1
    var_21 = var_6.__eq__(var_6)
    assert var_21 is True
    var_22 = var_5.__iter__()
    var_23 = var_16.values()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_23) == 1
    var_24 = var_8.__iter__()
    var_25 = var_9.messages()
    var_26 = var_23.__repr__()
    assert var_26 == 'ValuesView(BaseError(text=[Message(text="Message(text=None, code=\'custom\')", code=\'custom\', index=[\'ValidationResult(value=None)\'])], code=\'custom\'))'
    var_27 = var_23.__contains__(var_15)
    assert var_27 is False
    var_28 = var_23.__contains__(var_25)
    assert var_28 is False
    var_29 = var_5.__repr__()
    assert var_29 == 'ValidationResult(value=None)'
    var_30 = var_20.__contains__(var_0)
    assert var_30 is False
    module_0.ParseError()