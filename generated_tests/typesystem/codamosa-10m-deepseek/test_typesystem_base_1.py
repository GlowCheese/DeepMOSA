# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    module_0.ValidationError()

def test_case_1():
    var_0 = 'T\n\tQs&;FhOH8p'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_2():
    var_0 = "k|)(&\x0c')0!k"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "k|)(&\x0c')0!k"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = False
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, code=var_2, key=var_1, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_4.keys()

def test_case_4():
    var_0 = None
    var_1 = '<JL4G'
    var_2 = module_0.Message(text=var_1, code=var_0, key=var_0, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '<JL4G'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='<JL4G', code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_5():
    module_0.ParseError()

def test_case_6():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    module_0.ParseError(code=var_0, key=var_3, position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    module_0.ValidationError()

def test_case_9():
    var_0 = None
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'None'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='None')"
    var_5 = var_2.__hash__()
    assert var_5 == -758758451066905743

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = '5~i0Fj3'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_2.__getitem__(var_1)

def test_case_11():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = [var_4]
    var_9 = module_0.BaseError(code=var_3, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.values()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_10) == 1
    var_11 = var_9.__iter__()
    var_12 = var_9.messages()
    var_13 = var_9.__str__()
    var_14 = var_9.__repr__()
    assert var_14 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_15 = var_9.__eq__(var_13)
    assert var_15 is False
    var_16 = None
    var_17 = module_0.ValidationError(text=var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17) == 1
    var_18 = module_0.Message(text=var_16, code=var_3, key=var_15, index=var_3, position=var_15)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text is None
    assert var_18.code == 'custom'
    assert var_18.index == [False]
    assert var_18.start_position is False
    assert var_18.end_position is False
    var_19 = var_18.__repr__()
    assert var_19 == "Message(text=None, code='custom', index=[False], position=False)"
    var_20 = '9^4'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_20, key=var_9, index=var_15)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__iter__()
    module_0.ValidationError(key=var_2, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = "37'x3F'[8%GMD"
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_2) == 1
    var_3 = True
    module_0.ValidationError(key=var_3, position=var_1, messages=var_1)

def test_case_14():
    var_0 = '">='
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, code=var_1, key=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == 'BaseError(text=\'">=\', code=\'custom\')'
    var_4 = None
    var_5 = module_0.Position(var_0, var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == '">='
    assert var_5.column_no is None
    assert var_5.char_index == '">='
    var_6 = module_0.Message(text=var_0, code=var_0, position=var_5, start_position=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == '">='
    assert var_6.code == '">='
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = var_5.__eq__(var_4)
    assert var_7 is False
    var_8 = var_5.__eq__(var_4)
    assert var_8 is False

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = module_0.ParseError(text=var_3, code=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_4) == 1
    var_5 = var_4.__str__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__iter__()
    module_0.ParseError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_6 = var_2.__repr__()
    assert var_6 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_7 = module_0.ValidationError(text=var_6, code=var_4, position=var_0, messages=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8) == 1
    var_9 = var_8.__contains__(var_0)
    assert var_9 is False
    var_7.__getitem__(var_9)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='ValidationResult(value=None)')"
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = [var_4]
    var_8 = module_0.BaseError(code=var_3, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    var_10 = var_8.__iter__()
    var_11 = var_8.messages()
    var_12 = var_8.__repr__()
    assert var_12 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_13 = var_8.__eq__(var_5)
    assert var_13 is False
    var_14 = None
    var_15 = module_0.ValidationError(text=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = module_0.Message(text=var_14, code=var_3, key=var_13, index=var_3, position=var_13)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == 'custom'
    assert var_16.index == [False]
    assert var_16.start_position is False
    assert var_16.end_position is False
    var_17 = var_16.__repr__()
    assert var_17 == "Message(text=None, code='custom', index=[False], position=False)"
    var_18 = var_15.__len__()
    assert var_18 == 1
    var_19 = module_0.Message(text=var_13, index=var_1, position=var_3)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is False
    assert var_19.code == 'custom'
    assert f'{type(var_19.index).__module__}.{type(var_19.index).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = var_18.__repr__()
    assert var_20 == '1'
    module_0.ValidationError(key=var_18)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = [var_4]
    var_9 = module_0.BaseError(code=var_3, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    var_11 = var_9.__iter__()
    var_12 = var_9.__eq__(var_10)
    assert var_12 is False
    var_13 = None
    var_14 = module_0.ValidationError(text=var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = module_0.Message(text=var_13, code=var_3, key=var_12, index=var_3, position=var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text is None
    assert var_15.code == 'custom'
    assert var_15.index == [False]
    assert var_15.start_position is False
    assert var_15.end_position is False
    var_16 = var_15.__repr__()
    assert var_16 == "Message(text=None, code='custom', index=[False], position=False)"
    module_0.ValidationError(code=var_16)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = 3220
    var_4 = module_0.Position(var_1, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.column_no is None
    assert var_4.char_index == 3220
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'ValidationResult(value=None)'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='ValidationResult(value=None)')"
    var_9 = var_1.__bool__()
    assert var_9 is True
    var_10 = [var_6]
    var_11 = module_0.BaseError(code=var_0, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_6.__hash__()
    assert var_12 == 6673850753728265181
    var_13 = None
    var_14 = var_11.__eq__(var_13)
    assert var_14 is False
    var_15 = var_11.__iter__()
    var_16 = var_11.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_16.__iter__()
    var_18 = var_17.__eq__(var_0)
    module_0.ValidationError(text=var_16, messages=var_17)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_1.__bool__()
    assert var_6 is True
    module_0.ParseError(code=var_0, key=var_3, position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = 'R*vas$9-'
    var_3 = module_0.Message(text=var_2, start_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'R*vas$9-'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.end_position is None
    var_4 = var_1.__bool__()
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text='R*vas$9-', code='custom', start_position=ValidationResult(value=None), end_position=None)"
    module_0.ParseError(text=var_0, key=var_2, messages=var_0)

def test_case_22():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {'key': 'value'}
    assert var_3.error is None
    var_4 = var_3.__iter__()
    var_5 = list(var_4)
    var_6 = 'Error message'
    var_7 = module_0.Message(text=var_6, code=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error message'
    assert var_7.code == 'key'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = module_0.ValidationResult(error=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert f'{type(var_10.error).__module__}.{type(var_10.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10.error) == 1
    var_11 = var_10.__iter__()
    var_12 = list(var_11)
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = '|\t/yi4'
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_3.values()

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = [var_4]
    var_8 = module_0.BaseError(code=var_3, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    var_10 = var_8.__len__()
    assert var_10 == 1
    var_11 = var_8.__str__()
    var_12 = var_8.__eq__(var_0)
    assert var_12 is False
    var_13 = var_8.__len__()
    assert var_13 == 1
    module_0.ValidationError(code=var_0, position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = [var_4, var_4, var_4]
    var_9 = module_0.BaseError(code=var_3, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == "{'': None}"
    var_11 = None
    var_12 = var_9.__iter__()
    var_13 = var_9.__repr__()
    assert var_13 == "BaseError([Message(text=None, code='ValidationResult(value=None)'), Message(text=None, code='ValidationResult(value=None)'), Message(text=None, code='ValidationResult(value=None)')])"
    var_14 = var_9.__eq__(var_10)
    assert var_14 is False
    var_15 = None
    var_16 = module_0.ValidationError(text=var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16) == 1
    var_17 = module_0.Message(text=var_15, code=var_3, key=var_14, index=var_3, position=var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text is None
    assert var_17.code == 'custom'
    assert var_17.index == [False]
    assert var_17.start_position is False
    assert var_17.end_position is False
    var_18 = var_17.__repr__()
    assert var_18 == "Message(text=None, code='custom', index=[False], position=False)"
    var_19 = module_0.ValidationError(text=var_3, position=var_11, messages=var_8)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19) == 1
    module_0.ValidationError(code=var_3, key=var_6)

def test_case_26():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_3, messages=var_8)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = [var_4]
    var_8 = module_0.BaseError(code=var_3, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    var_10 = var_8.__iter__()
    var_11 = var_8.__str__()
    var_12 = var_8.__repr__()
    assert var_12 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_13 = var_8.__eq__(var_11)
    assert var_13 is False
    var_14 = None
    var_15 = var_8.values()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_15) == 1
    var_16 = var_13.__eq__(var_14)
    module_0.ValidationError(position=var_16, messages=var_16)

def test_case_28():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'ValidationResult(value=None)'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text=None, code='ValidationResult(value=None)')"
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = [var_3]
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_4, messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = [var_4]
    var_9 = module_0.BaseError(code=var_3, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__iter__()
    var_11 = var_9.messages()
    var_12 = var_9.__str__()
    var_13 = var_9.__repr__()
    assert var_13 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_14 = var_9.__eq__(var_12)
    assert var_14 is False
    var_15 = module_0.ValidationError(text=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = module_0.Message(text=var_9, code=var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_16.text).__module__}.{type(var_16.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16.text) == 1
    assert var_16.code is False
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = var_16.__repr__()
    assert var_17 == "Message(text=BaseError(text=None, code='ValidationResult(value=None)'), code=False)"
    var_18 = var_15.__len__()
    assert var_18 == 1
    var_19 = module_0.Message(text=var_3, position=var_0, end_position=var_3)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is None
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = var_16.__repr__()
    assert var_20 == "Message(text=BaseError(text=None, code='ValidationResult(value=None)'), code=False)"
    module_0.ValidationError(code=var_14, key=var_18)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = [var_4]
    var_9 = module_0.BaseError(code=var_3, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    var_11 = var_9.__hash__()
    assert var_11 == 2849844545640766516
    var_12 = var_9.__iter__()
    var_13 = var_9.messages()
    var_14 = var_9.__str__()
    var_15 = var_9.__repr__()
    assert var_15 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_16 = var_9.__eq__(var_14)
    assert var_16 is False
    var_17 = None
    var_18 = module_0.ValidationError(text=var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18) == 1
    var_19 = module_0.Message(text=var_17, code=var_3, key=var_16, index=var_3, position=var_16)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is None
    assert var_19.code == 'custom'
    assert var_19.index == [False]
    assert var_19.start_position is False
    assert var_19.end_position is False
    var_20 = var_1.__iter__()
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, code=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_3)
    assert var_5 is False
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_1.__bool__()
    assert var_7 is True
    var_8 = [var_4]
    var_9 = module_0.BaseError(code=var_3, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    var_11 = None
    var_12 = var_9.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_9.messages(add_prefix=var_2)
    var_14 = var_9.__str__()
    var_15 = var_9.__repr__()
    assert var_15 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_16 = var_9.__eq__(var_11)
    assert var_16 is False
    var_17 = module_0.Message(text=var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text is None
    assert var_17.code == 'custom'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = var_4.__repr__()
    assert var_18 == "Message(text=None, code='ValidationResult(value=None)')"
    module_0.ValidationError(text=var_11, key=var_0)

def test_case_32():
    var_0 = None
    var_1 = var_0.__repr__()
    assert var_1 == 'None'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'None'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='None')"

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = True
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is True
    assert var_2.char_index is True
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=True, char_index=True)'
    var_4 = None
    module_0.ValidationError(code=var_4)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'ValidationResult(value=None)'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text=None, code='ValidationResult(value=None)')"
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = module_0.BaseError(text=var_2, key=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_7.__str__()
    assert var_8 == "{True: 'ValidationResult(value=None)'}"
    var_9 = None
    var_10 = var_7.__eq__(var_9)
    assert var_10 is False
    var_11 = var_7.__iter__()
    var_12 = var_7.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_7.__repr__()
    assert var_13 == "BaseError([Message(text='ValidationResult(value=None)', code='custom', index=[True])])"
    var_14 = var_7.__repr__()
    assert var_14 == "BaseError([Message(text='ValidationResult(value=None)', code='custom', index=[True])])"
    var_15 = var_12.__len__()
    assert var_15 == 1
    var_16 = None
    var_17 = '[_<|&Qq\x0b;f"A'
    var_18 = module_0.ValidationError(text=var_14, code=var_16, position=var_16, messages=var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18) == 1
    var_19 = 'y#?+.g"k4'
    var_20 = module_0.Message(text=var_19, end_position=var_17)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'y#?+.g"k4'
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position == '[_<|&Qq\x0b;f"A'
    var_21 = var_20.__repr__()
    assert var_21 == 'Message(text=\'y#?+.g"k4\', code=\'custom\')'
    var_22 = var_1.__iter__()
    module_0.ValidationError(code=var_5, key=var_15)

def test_case_35():
    var_0 = None
    var_1 = None
    var_2 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, code=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'ValidationResult(value=None)'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='ValidationResult(value=None)')"
    var_7 = var_2.__bool__()
    assert var_7 is True
    var_8 = [var_4]
    var_9 = module_0.BaseError(code=var_0, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    var_11 = None
    var_12 = var_9.__eq__(var_11)
    assert var_12 is False
    var_13 = var_9.__iter__()
    var_14 = var_9.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = var_9.__repr__()
    assert var_15 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_16 = var_9.__contains__(var_0)
    assert var_16 is False
    var_17 = var_2.__bool__()
    assert var_17 is True
    var_18 = var_14.__len__()
    assert var_18 == 1
    var_19 = var_14.__eq__(var_11)
    var_20 = None
    var_21 = module_0.ValidationError(text=var_15, position=var_8)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_21) == 1
    var_22 = var_21.messages()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_20, code=var_0, index=var_19, position=var_14, end_position=var_16)

def test_case_36():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'ValidationResult(value=None)'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text=None, code='ValidationResult(value=None)')"
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = [var_3]
    var_8 = module_0.BaseError(code=var_0, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    var_10 = var_8.__eq__(var_0)
    assert var_10 is False
    var_11 = var_8.__iter__()
    var_12 = var_8.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_8.__contains__(var_0)
    assert var_13 is False
    var_14 = var_8.__repr__()
    assert var_14 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_15 = var_12.__len__()
    assert var_15 == 1
    var_16 = var_15.__eq__(var_13)
    assert var_16 is False
    var_17 = None
    var_18 = '_<|&Qq\x0b;f"'
    var_19 = var_3.__repr__()
    assert var_19 == "Message(text=None, code='ValidationResult(value=None)')"
    var_20 = module_0.ValidationError(text=var_18, code=var_0, key=var_15)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20) == 1
    var_21 = module_0.Message(text=var_8, code=var_16, index=var_15)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_21.text).__module__}.{type(var_21.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_21.text) == 1
    assert var_21.code is False
    assert var_21.index == 1
    assert var_21.start_position is None
    assert var_21.end_position is None
    var_22 = var_15.__repr__()
    assert var_22 == '1'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, code=var_17, index=var_15, position=var_21, start_position=var_16)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'ValidationResult(value=None)'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text=None, code='ValidationResult(value=None)')"
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = [var_3]
    var_8 = module_0.BaseError(code=var_0, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    var_10 = None
    var_11 = var_8.__eq__(var_10)
    assert var_11 is False
    var_12 = var_8.__iter__()
    var_13 = var_8.keys()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13) == 1
    var_14 = var_8.__repr__()
    assert var_14 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_15 = var_8.__contains__(var_0)
    assert var_15 is False
    var_16 = var_1.__bool__()
    assert var_16 is True
    var_17 = var_13.__len__()
    assert var_17 == 1
    var_18 = var_13.__eq__(var_10)
    var_19 = None
    var_20 = module_0.ValidationError(text=var_14, position=var_7)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20) == 1
    var_21 = var_20.messages()
    var_22 = 'Ar.r8'
    var_23 = module_0.Message(text=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'Ar.r8'
    assert var_23.code == 'custom'
    assert var_23.index == []
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = var_17.__repr__()
    assert var_24 == '1'
    var_25 = module_0.Message(text=var_19, index=var_18, end_position=var_18)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text is None
    assert var_25.code == 'custom'
    assert f'{type(var_25.index).__module__}.{type(var_25.index).__qualname__}' == 'builtins.NotImplementedType'
    assert var_25.start_position is None
    assert f'{type(var_25.end_position).__module__}.{type(var_25.end_position).__qualname__}' == 'builtins.NotImplementedType'
    var_26 = module_0.ValidationResult(value=var_0, error=var_17)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_26.value is None
    assert var_26.error == 1
    var_27 = var_26.__repr__()
    assert var_27 == 'ValidationResult(error=1)'
    var_28 = var_18.__repr__()
    assert var_28 == 'NotImplemented'
    module_0.ValidationError(code=var_17, key=var_11, position=var_0, messages=var_0)

def test_case_38():
    var_0 = None
    var_1 = '[_<|&Qq\x0b;f"A'
    var_2 = '[D\\n\ts|C0'
    var_3 = module_0.Message(text=var_2, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '[D\\n\ts|C0'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value == '[_<|&Qq\x0b;f"A'
    assert var_4.error is None

def test_case_39():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {'key': 'value'}
    assert var_3.error is None
    var_4 = var_3.__iter__()
    var_5 = list(var_4)
    var_6 = 'Error message'
    var_7 = 'error_code'
    var_8 = module_0.Message(text=var_6, code=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error message'
    assert var_8.code == 'error_code'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = module_0.ValidationResult(error=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert f'{type(var_11.error).__module__}.{type(var_11.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11.error) == 1
    var_12 = var_11.__iter__()
    var_13 = list(var_12)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_2, error=var_10)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = 3220
    var_4 = module_0.Position(var_1, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.column_no is None
    assert var_4.char_index == 3220
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'ValidationResult(value=None)'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__eq__(var_6)
    assert var_7 is True
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text=None, code='ValidationResult(value=None)')"
    var_9 = var_1.__bool__()
    assert var_9 is True
    var_10 = [var_6]
    var_11 = module_0.BaseError(code=var_0, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_11.__str__()
    var_13 = var_6.__hash__()
    assert var_13 == 6673850753728265181
    var_14 = None
    var_15 = var_11.__eq__(var_14)
    assert var_15 is False
    var_16 = var_11.__iter__()
    var_17 = var_11.keys()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_17) == 1
    var_18 = var_11.__repr__()
    assert var_18 == "BaseError(text=None, code='ValidationResult(value=None)')"
    var_19 = var_11.__contains__(var_0)
    assert var_19 is False
    var_20 = var_17.__repr__()
    assert var_20 == "KeysView(BaseError(text=None, code='ValidationResult(value=None)'))"
    var_21 = var_11.__contains__(var_0)
    assert var_21 is False
    var_22 = None
    var_23 = module_0.ValidationError(text=var_18, position=var_10)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_23) == 1
    var_24 = var_23.messages()
    var_25 = module_0.Message(text=var_14)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text is None
    assert var_25.code == 'custom'
    assert var_25.index == []
    assert var_25.start_position is None
    assert var_25.end_position is None
    var_26 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_26.value is None
    assert var_26.error is None
    var_27 = var_11.__eq__(var_23)
    assert var_27 is False
    var_28 = var_6.__repr__()
    assert var_28 == "Message(text=None, code='ValidationResult(value=None)')"
    var_29 = var_23.__contains__(var_22)
    assert var_29 is False
    var_30 = var_23.keys()
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_30) == 1
    var_31 = {}
    var_27.get(var_0, var_31)

def test_case_41():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = 3186
    var_4 = module_0.Position(var_1, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_4.line_no).__module__}.{type(var_4.line_no).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.column_no is None
    assert var_4.char_index == 3186
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_1.__bool__()
    assert var_6 is True
    var_7 = var_4.__eq__(var_4)
    assert var_7 is True

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 2
    assert var_4.char_index == 3
    var_5 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 2
    assert var_5.char_index == 3
    var_6 = 4
    var_7 = module_0.Position(var_0, var_1, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 2
    assert var_7.char_index == 4
    var_8 = var_5 == var_7
    var_9 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 1
    assert var_9.column_no == 2
    assert var_9.char_index == 3
    var_10 = module_0.Position(var_0, var_2, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 3
    assert var_10.char_index == 3
    var_11 = var_9 == var_10
    var_8.__contains__(var_11)

def test_case_43():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 2
    assert var_4.char_index == 3
    var_5 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 2
    assert var_5.char_index == 3
    var_6 = 4
    var_7 = module_0.Position(var_0, var_1, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 2
    assert var_7.char_index == 4
    var_8 = var_5 == var_7
    var_9 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 1
    assert var_9.column_no == 2
    assert var_9.char_index == 3
    var_10 = module_0.Position(var_0, var_2, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 3
    assert var_10.char_index == 3
    var_11 = var_9 == var_10
    var_12 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == 1
    assert var_12.column_no == 2
    assert var_12.char_index == 3
    var_13 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 2
    assert var_13.column_no == 2
    assert var_13.char_index == 3
    var_14 = var_12 == var_13
    var_15 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no == 1
    assert var_15.column_no == 2
    assert var_15.char_index == 3
    var_16 = 'not a Position'
    var_17 = var_15 == var_16

def test_case_44():
    var_0 = 'error message'
    var_1 = 'error_code'
    var_2 = 'key'
    var_3 = module_0.BaseError(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'error message'
    assert var_4.code == 'error_code'
    assert var_4.index == ['key']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'error1'
    var_6 = 'code1'
    var_7 = 'key1'
    var_8 = module_0.Message(text=var_5, code=var_6, key=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'error1'
    assert var_8.code == 'code1'
    assert var_8.index == ['key1']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = 'error2'
    var_10 = 'code2'
    var_11 = 'key2'
    var_12 = module_0.Message(text=var_9, code=var_10, key=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'error2'
    assert var_12.code == 'code2'
    assert var_12.index == ['key2']
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = [var_8, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 2
    var_15 = 'subkey1'
    var_16 = [var_7, var_15]
    var_17 = module_0.Message(text=var_5, code=var_6, index=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'error1'
    assert var_17.code == 'code1'
    assert var_17.index == ['key1', 'subkey1']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = 'subkey2'
    var_19 = [var_11, var_18]
    var_20 = module_0.Message(text=var_9, code=var_10, index=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'error2'
    assert var_20.code == 'code2'
    assert var_20.index == ['key2', 'subkey2']
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = [var_17, var_20]
    var_22 = module_0.BaseError(messages=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_22) == 2
    var_23 = 1
    var_24 = 0
    var_25 = module_0.Position(var_23, var_23, var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Position'
    assert var_25.line_no == 1
    assert var_25.column_no == 1
    assert var_25.char_index == 0
    var_26 = module_0.BaseError(text=var_0, code=var_1, position=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_26) == 1
    var_27 = module_0.Message(text=var_0, code=var_1, position=var_25)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Message'
    assert var_27.text == 'error message'
    assert var_27.code == 'error_code'
    assert var_27.index == []
    assert f'{type(var_27.start_position).__module__}.{type(var_27.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_27.end_position).__module__}.{type(var_27.end_position).__qualname__}' == 'typesystem.base.Position'