# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = '$SI\rQec_C"'
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '$SI\rQec_C"'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = '9Error'
    var_4 = None
    var_5 = module_0.ParseError(text=var_3, code=var_3, key=var_4, position=var_4, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_5.__hash__()
    assert var_6 == 6662897838330892443
    var_7 = module_0.Message(text=var_3, code=var_3, position=var_4, start_position=var_4, end_position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '9Error'
    assert var_7.code == '9Error'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = bool(var_7 == var_7)
    assert var_8 is True

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
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None

def test_case_6():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = "'/c9\r~-KueS|QN*IX"
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Position(var_3, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = module_0.Message(text=var_4, code=var_4, index=var_2, start_position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == "'/c9\r~-KueS|QN*IX"
    assert var_7.code == "'/c9\r~-KueS|QN*IX"
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_7.end_position is None
    var_8 = var_7.__repr__()
    assert var_8 == 'Message(text="\'/c9\\r~-KueS|QN*IX", code="\'/c9\\r~-KueS|QN*IX", start_position=Position(line_no=Message(text=None, code=\'custom\'), column_no=None, char_index=False), end_position=None)'
    var_9 = var_1.__repr__()
    assert var_9 == "Message(text=None, code='custom')"

def test_case_7():
    var_0 = '+3}*=?U,|L"a/_)D'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__hash__()
    assert var_3 == -2969124648911349233

def test_case_8():
    var_0 = ' t_`BU!cB<G'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1

def test_case_9():
    var_0 = "'\x0brfVk"
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "'\x0brfVk"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, index=var_1, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "'\x0brfVk"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.Message'
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False

def test_case_10():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_2) == 1
    var_3 = var_2.__hash__()
    var_4 = var_2.__iter__()
    var_5 = None
    var_6 = var_1.__eq__(var_5)
    assert var_6 is False
    var_7 = module_0.Position(var_2, var_4, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_7.line_no) == 1
    assert f'{type(var_7.column_no).__module__}.{type(var_7.column_no).__qualname__}' == 'builtins.generator'
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'builtins.generator'
    var_8 = var_2.__repr__()
    assert var_8 == "ValuesView(BaseError(text='.s;', code='custom'))"
    var_9 = var_4.__repr__()

def test_case_11():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'Error'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "ValuesView(ParseError(text='Error', code='Error'))"

def test_case_12():
    var_0 = 'Jax36jUN`\rr)U]+r'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()

def test_case_13():
    var_0 = 'Jax36jUN`\rr)U]+r'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ParseError(text='Jax36jUN`\\rr)U]+r', code='Jax36jUN`\\rr)U]+r')"
    var_4 = var_2.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_4) == 1
    var_5 = bool(var_4 == var_4)
    assert var_5 is True

def test_case_14():
    var_0 = 'Jax36jUN`\rr)U]+r'
    var_1 = None
    var_2 = -1604
    var_3 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__bool__()
    assert var_4 is True
    var_5 = -1422
    var_6 = module_0.Position(var_2, var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == -1604
    assert var_6.column_no is None
    assert var_6.char_index == -1422
    var_7 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_7.__repr__()
    assert var_8 == "ParseError(text='Jax36jUN`\\rr)U]+r', code='Jax36jUN`\\rr)U]+r')"
    var_9 = var_7.__iter__()
    var_10 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'Jax36jUN`\rr)U]+r'
    assert var_10.code == 'Jax36jUN`\rr)U]+r'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = bool(var_10 == var_10)
    assert var_11 is True

def test_case_15():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error'
    assert var_1.code == 'Error'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 == var_1)
    assert var_2 is True

def test_case_16():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = [var_1]
    var_3 = module_0.ValidationError(messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__str__()

def test_case_17():
    var_0 = ' t_`BU!cB<G'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, code=var_0, key=var_1, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_2.__len__()
    assert var_4 == 1

def test_case_18():
    var_0 = '*)}{/rUcgfUkdI-\r'
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '*)}{/rUcgfUkdI-\r'
    assert var_1.code == '*)}{/rUcgfUkdI-\r'
    assert var_1.index == ['*)}{/rUcgfUkdI-\r']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = bool(var_1 == var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_2.text).__module__}.{type(var_2.text).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert f'{type(var_2.start_position).__module__}.{type(var_2.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text=ValidationResult(value=None), code='custom', index=[ValidationResult(value=None)], start_position=ValidationResult(value=None), end_position=None)"
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = module_0.Message(text=var_1, code=var_1, position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__hash__()
    var_7 = var_5.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_2.__eq__(var_1)
    assert var_8 is False
    var_9 = None
    var_10 = module_0.Message(text=var_9, code=var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == "Message(text=None, code='custom')"
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_0.__repr__()
    assert var_11 == 'ValidationResult(value=None)'
    var_12 = var_0.__repr__()
    assert var_12 == 'ValidationResult(value=None)'
    var_13 = module_0.BaseError(text=var_7, code=var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_0.__bool__()
    assert var_14 is True
    var_15 = var_13.__iter__()
    var_16 = var_10.__hash__()
    assert var_16 == -2629022533388233681
    module_0.ValidationError()

def test_case_20():
    var_0 = 'b|vTI}\ny`wbD{dY>2'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'X.rJ`|yid]{nl\n8t=.'
    var_3 = module_0.Message(text=var_2, index=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'X.rJ`|yid]{nl\n8t=.'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = None
    var_7 = var_1.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.ValidationResult(value=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = [var_3, var_3, var_1, var_1]
    module_0.ValidationError(position=var_3, messages=var_9)

def test_case_22():
    var_0 = 1
    var_1 = 5
    var_2 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == 5
    assert var_2.error is None
    var_3 = 10
    var_4 = module_0.Position(var_0, var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 5
    assert var_4.char_index == 10
    var_5 = 2
    var_6 = 6
    var_7 = 11
    var_8 = module_0.Position(var_5, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == 2
    assert var_8.column_no == 6
    assert var_8.char_index == 11
    var_9 = bool(not var_4 == var_8)
    assert var_9 is True

def test_case_23():
    var_0 = 'Egror'
    var_1 = 'error_code'
    var_2 = 'field1'
    var_3 = [var_1, var_2, var_1]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Egror'
    assert var_4.code == 'error_code'
    assert var_4.index == ['error_code', 'field1', 'error_code']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_1]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Egror'
    assert var_6.code == 'error_code'
    assert var_6.index == ['error_code']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = bool(not var_4 == var_6)
    assert var_7 is True

def test_case_24():
    var_0 = 'S8[@FBhrOL\x0bc:C)-'
    var_1 = module_0.Message(text=var_0, code=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'S8[@FBhrOL\x0bc:C)-'
    assert var_1.code == 'S8[@FBhrOL\x0bc:C)-'
    assert var_1.index == []
    assert var_1.start_position == 'S8[@FBhrOL\x0bc:C)-'
    assert var_1.end_position == 'S8[@FBhrOL\x0bc:C)-'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.ParseError(text=var_1, key=var_5, messages=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == '{False: ValidationResult(value=None)}'
    var_8 = var_6.__iter__()
    var_9 = var_8.__iter__()
    var_10 = var_6.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == "ItemsView(ParseError([Message(text=ValidationResult(value=None), code='custom', index=[False])]))"
    module_0.BaseError(messages=var_8)

def test_case_26():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = None
    var_5 = None
    var_6 = module_0.Message(text=var_5, key=var_0, index=var_0, position=var_4, start_position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = "'/c9\r~-KueS|QN*IX"
    var_8 = 2139
    var_9 = var_6.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.Position(var_6, var_4, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_10.line_no).__module__}.{type(var_10.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_10.column_no is None
    assert var_10.char_index == 2139
    var_11 = module_0.Message(text=var_7, code=var_7, index=var_4, start_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == "'/c9\r~-KueS|QN*IX"
    assert var_11.code == "'/c9\r~-KueS|QN*IX"
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_11.end_position is None
    var_12 = var_11.__repr__()
    assert var_12 == 'Message(text="\'/c9\\r~-KueS|QN*IX", code="\'/c9\\r~-KueS|QN*IX", start_position=Position(line_no=Message(text=None, code=\'custom\'), column_no=None, char_index=2139), end_position=None)'
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_10, error=var_8)

def test_case_27():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_2, key=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = module_0.Position(var_3, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_5.column_no is None
    assert var_5.char_index is False
    var_6 = module_0.Message(text=var_1, code=var_1, index=var_2, start_position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'ValidationResult(value=None)'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_6.end_position is None
    var_7 = var_6.__repr__()
    assert var_7 == "Message(text='ValidationResult(value=None)', code='ValidationResult(value=None)', start_position=Position(line_no=Message(text=None, code='custom'), column_no=None, char_index=False), end_position=None)"
    var_8 = module_0.ParseError(text=var_0, key=var_4, messages=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == '{False: ValidationResult(value=None)}'
    var_10 = module_0.ValidationError(text=var_1, position=var_2, messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_8.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = var_11.__repr__()
    assert var_12 == "ItemsView(ParseError([Message(text=ValidationResult(value=None), code='custom', index=[False])]))"
    var_13 = var_11.__eq__(var_10)
    var_14 = var_12.__iter__()
    var_15 = module_0.BaseError(text=var_11, position=var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = var_12.__hash__()
    assert var_16 == 3416982554275162438
    var_17 = var_8.__eq__(var_10)
    assert var_17 is False
    var_18 = var_8.__iter__()
    var_19 = var_8.messages()
    var_20 = module_0.Position(var_17, var_17, var_17)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no is False
    assert var_20.column_no is False
    assert var_20.char_index is False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_4, position=var_10, messages=var_11)

def test_case_28():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = None
    var_4 = module_0.Message(text=var_3, key=var_3, index=var_3, position=var_3, end_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = True
    var_7 = module_0.Position(var_0, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is True
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'typesystem.base.Message'
    var_8 = 'qE'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_8, key=var_2, index=var_2, start_position=var_0)

def test_case_29():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_2, key=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = "'/c9\r~-KueS|QN*IX"
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_6 = module_0.Position(var_3, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = module_0.Message(text=var_4, code=var_4, index=var_2, start_position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == "'/c9\r~-KueS|QN*IX"
    assert var_7.code == "'/c9\r~-KueS|QN*IX"
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_7.end_position is None
    var_8 = var_7.__repr__()
    assert var_8 == 'Message(text="\'/c9\\r~-KueS|QN*IX", code="\'/c9\\r~-KueS|QN*IX", start_position=Position(line_no=Message(text=None, code=\'custom\'), column_no=None, char_index=False), end_position=None)'
    var_9 = module_0.ParseError(text=var_0, key=var_5, messages=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == '{False: ValidationResult(value=None)}'
    var_11 = var_9.__iter__()
    var_12 = var_11.__iter__()
    var_13 = var_9.items()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_13) == 1
    var_14 = var_13.__repr__()
    assert var_14 == "ItemsView(ParseError([Message(text=ValidationResult(value=None), code='custom', index=[False])]))"
    var_15 = var_13.__eq__(var_12)
    var_16 = var_11.__iter__()
    var_17 = module_0.BaseError(text=var_13, position=var_0)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = var_9.__eq__(var_12)
    assert var_18 is False
    var_19 = "*w6H0%'"
    var_20 = module_0.BaseError(text=var_19, code=var_18, key=var_12, position=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = var_13.__iter__()
    var_22 = var_17.messages()
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_13, position=var_21, messages=var_13)

def test_case_30():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = None
    var_5 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = "'/c9\r~-KueS|QN*IX"
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Position(var_5, var_4, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.line_no).__module__}.{type(var_8.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_8.column_no is None
    assert var_8.char_index is False
    var_9 = module_0.Message(text=var_6, code=var_6, index=var_4, start_position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == "'/c9\r~-KueS|QN*IX"
    assert var_9.code == "'/c9\r~-KueS|QN*IX"
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_9.end_position is None
    var_10 = var_9.__repr__()
    assert var_10 == 'Message(text="\'/c9\\r~-KueS|QN*IX", code="\'/c9\\r~-KueS|QN*IX", start_position=Position(line_no=Message(text=None, code=\'custom\'), column_no=None, char_index=False), end_position=None)'
    var_11 = None
    var_12 = module_0.ParseError(text=var_1, key=var_7, messages=var_4)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = None
    var_14 = module_0.ParseError(text=var_10, code=var_11, position=var_4)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_12.__iter__()
    var_16 = var_12.__iter__()
    var_17 = var_14.items()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17) == 1
    var_18 = var_15.__repr__()
    var_19 = var_17.__iter__()
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_13, key=var_2, messages=var_17)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = None
    var_5 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = "'/c9\r~-KueS|QN*IX"
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.ParseError(text=var_1, key=var_7, messages=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == '{False: ValidationResult(value=None)}'
    var_10 = var_6.__iter__()
    var_10.items()

def test_case_32():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_2, key=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_6 = module_0.Position(var_3, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = module_0.Message(text=var_1, code=var_1, index=var_2, start_position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'ValidationResult(value=None)'
    assert var_7.code == 'ValidationResult(value=None)'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_7.end_position is None
    var_8 = var_7.__repr__()
    assert var_8 == "Message(text='ValidationResult(value=None)', code='ValidationResult(value=None)', start_position=Position(line_no=Message(text=None, code='custom'), column_no=None, char_index=False), end_position=None)"
    var_9 = module_0.ParseError(text=var_0, key=var_5, messages=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == '{False: ValidationResult(value=None)}'
    var_11 = var_9.__iter__()
    var_12 = var_11.__iter__()
    var_13 = var_6.__repr__()
    assert var_13 == "Position(line_no=Message(text=None, code='custom'), column_no=None, char_index=False)"
    var_14 = var_2.__eq__(var_12)
    var_15 = var_9.__eq__(var_12)
    assert var_15 is False
    var_16 = module_0.BaseError(text=var_15, code=var_15, messages=var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = var_12.__iter__()
    var_18 = var_16.messages(add_prefix=var_2)
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_12, key=var_11, messages=var_12)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = None
    var_5 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = 1005
    var_7 = module_0.Position(var_4, var_0, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is None
    assert var_7.char_index == 1005
    var_8 = module_0.Message(text=var_4, index=var_0, position=var_7, end_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = var_8.__repr__()
    assert var_9 == "Message(text=None, code='custom', position=Position(line_no=None, column_no=None, char_index=1005))"
    module_0.ParseError(code=var_4)

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
    var_2 = None
    var_3 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = "'/c9\r~-KueS|QN*IX"
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Position(var_3, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = module_0.Message(text=var_4, code=var_4, index=var_2, start_position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == "'/c9\r~-KueS|QN*IX"
    assert var_7.code == "'/c9\r~-KueS|QN*IX"
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_7.end_position is None
    var_8 = var_7.__repr__()
    assert var_8 == 'Message(text="\'/c9\\r~-KueS|QN*IX", code="\'/c9\\r~-KueS|QN*IX", start_position=Position(line_no=Message(text=None, code=\'custom\'), column_no=None, char_index=False), end_position=None)'
    var_9 = ()
    module_0.ValidationError(text=var_2, messages=var_9)

def test_case_35():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = None
    var_5 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = "'/c9\r~-KueS|QN*IX"
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Position(var_5, var_4, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.line_no).__module__}.{type(var_8.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_8.column_no is None
    assert var_8.char_index is False
    var_9 = 248
    var_10 = module_0.Position(var_9, var_4, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 248
    assert var_10.column_no is None
    assert var_10.char_index is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_4, code=var_2, key=var_0, index=var_6, position=var_10, end_position=var_10)

def test_case_36():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 5
    assert var_3.char_index == 10
    var_4 = module_0.Position(var_0, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 10
    assert var_4.char_index == 10
    var_5 = bool(not var_3 == var_4)
    assert var_5 is True

def test_case_37():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_2, key=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = "'/c9\r~-ueS|QNh*I"
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_6 = True
    var_7 = module_0.Position(var_5, var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is False
    assert var_7.column_no is True
    assert var_7.char_index is None
    var_8 = ')DE/U$%?UUmPM='
    with pytest.raises(AssertionError):
        module_0.Message(text=var_8, code=var_4, position=var_8, start_position=var_7)

def test_case_38():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_2, key=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = "'/c9\r~-KueS|QN*IX"
    var_5 = var_3.__eq__(var_2)
    assert var_5 is False
    var_6 = module_0.Position(var_3, var_2, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Message'
    assert var_6.column_no is None
    assert var_6.char_index is False
    var_7 = module_0.Message(text=var_4, code=var_4, index=var_2, start_position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == "'/c9\r~-KueS|QN*IX"
    assert var_7.code == "'/c9\r~-KueS|QN*IX"
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_7.end_position is None
    var_8 = var_7.__repr__()
    assert var_8 == 'Message(text="\'/c9\\r~-KueS|QN*IX", code="\'/c9\\r~-KueS|QN*IX", start_position=Position(line_no=Message(text=None, code=\'custom\'), column_no=None, char_index=False), end_position=None)'
    var_9 = None
    var_10 = module_0.ParseError(text=var_0, key=var_5, messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.__str__()
    assert var_11 == '{False: ValidationResult(value=None)}'
    var_12 = var_10.__iter__()
    var_13 = var_12.__iter__()
    var_14 = var_10.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14) == 1
    var_15 = var_14.__repr__()
    assert var_15 == "ItemsView(ParseError([Message(text=ValidationResult(value=None), code='custom', index=[False])]))"
    var_16 = var_14.__eq__(var_13)
    var_17 = var_12.__iter__()
    var_18 = module_0.BaseError(text=var_14, position=var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_18) == 1
    var_19 = var_10.__eq__(var_9)
    assert var_19 is False
    var_20 = module_0.BaseError(text=var_19, code=var_14, messages=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20) == 1
    var_21 = var_10.messages(add_prefix=var_15)
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_9)

@pytest.mark.xfail(strict=True)
def test_case_39():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_2, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = module_0.Position(var_4, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is False
    var_6 = module_0.Message(text=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__repr__()
    assert var_7 == "Message(text='ValidationResult(value=None)', code='custom')"
    var_8 = [var_6, var_3]
    var_9 = module_0.ParseError(position=var_2, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == "{'': None}"
    var_11 = var_9.__iter__()
    module_0.ValidationError(position=var_2, messages=var_11)

def test_case_40():
    var_0 = 'Error'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Error'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'viZ[:kM*s^~s}Y7'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'viZ[:kM*s^~s}Y7'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_1 == var_3)

def test_case_41():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 5
    assert var_3.char_index == 10
    var_4 = 2
    var_5 = module_0.Position(var_4, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 2
    assert var_5.column_no == 5
    assert var_5.char_index == 10
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_42():
    var_0 = '{:bm1hkz'
    var_1 = 'min_length'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == '{:bm1hkz'
    assert var_6.code == 'min_length'
    assert var_6.index == ['users', 0, 'name']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Invalid email'
    var_8 = 'invalid_email'
    var_9 = 'email'
    var_10 = [var_2, var_3, var_9]
    var_11 = module_0.Message(text=var_7, code=var_8, index=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'Invalid email'
    assert var_11.code == 'invalid_email'
    assert var_11.index == ['users', 0, 'email']
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = [var_6, var_11]
    var_13 = module_0.BaseError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_13._message_dict
    var_15 = bool(var_13._message_dict == {'users': {0: {'name': 'Too short', 'email': 'Invalid email'}}})

def test_case_43():
    var_0 = 'Invalid input'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = repr(var_2)
    var_4 = bool('ValidationResult' in var_3)
    assert var_4 is True
    var_5 = bool('error=' in var_3)
    assert var_5 is True
    var_6 = bool('Invalid input' in var_3)
    assert var_6 is True

def test_case_44():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = repr(var_1)
    var_3 = bool('ValidationResult' in var_1)
    var_4 = bool('value=' in var_2)
    assert var_4 is True
    var_5 = bool('None' in var_2)
    assert var_5 is True

def test_case_45():
    var_0 = 'Error 2'
    var_1 = 'error2'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error 2'
    assert var_2.code == 'error2'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = [var_2, var_2]
    var_4 = module_0.BaseError(messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = repr(var_4)
    var_6 = bool('BaseError([Message' in var_5)
    assert var_6 is True
    var_7 = bool('Error 2' in var_5)
    assert var_7 is True