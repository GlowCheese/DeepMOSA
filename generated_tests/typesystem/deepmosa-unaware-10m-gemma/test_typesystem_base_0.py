# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.BaseError(text=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1

def test_case_1():
    var_0 = '"ZOg[&(rQuT(&\\\\J'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, position=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '"ZOg[&(rQuT(&\\\\J'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None

def test_case_2():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.Message(text=var_1, code=var_1, key=var_1, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'ValidationResult(value=None)'
    assert var_3.code == 'ValidationResult(value=None)'
    assert var_3.index == ['ValidationResult(value=None)']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    module_0.ValidationError(code=var_0, key=var_0)

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_5():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_2.__contains__(var_1)

def test_case_7():
    var_0 = None
    var_1 = 2641
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == 2641
    assert var_2.char_index == 2641
    var_3 = "T1AVM]E&^5c'>JT;"
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = None
    var_6 = module_0.BaseError(text=var_3, code=var_5, key=var_5, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__iter__()

def test_case_8():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = var_0.__bool__()
    assert var_2 is True
    module_0.ValidationError()

def test_case_10():
    var_0 = None
    var_1 = '%'
    var_2 = module_0.ValidationError(text=var_1, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_3.__repr__()
    assert var_4 == "[Message(text='%', code='custom')]"
    var_5 = var_2.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "ItemsView(ValidationError(text='%', code='custom'))"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_2, position=var_2)

def test_case_11():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.__str__()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is None

def test_case_12():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None

def test_case_13():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.BaseError(text=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__repr__()
    assert var_4 == "BaseError(text=Position(line_no=None, column_no=None, char_index=None), code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 2641
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == 2641
    assert var_2.char_index == 2641
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = ''
    var_5 = module_0.ParseError(text=var_4, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_5.__contains__(var_0)
    assert var_6 is False
    var_7 = module_0.Message(text=var_4, key=var_4, index=var_0, position=var_6, start_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == ''
    assert var_7.code == 'custom'
    assert var_7.index == ['']
    assert var_7.start_position is False
    assert var_7.end_position is False
    var_8 = module_0.Position(var_0, var_6, var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is False
    assert var_8.char_index is None
    var_6.keys()

def test_case_15():
    var_0 = None
    var_1 = True
    var_2 = module_0.Position(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is True
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=True, column_no=None, char_index=None)'
    var_5 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0, messages=var_5)

def test_case_16():
    var_0 = None
    var_1 = '%'
    var_2 = module_0.ValidationError(text=var_1, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_3.__repr__()
    assert var_4 == "[Message(text='%', code='custom')]"
    var_5 = False
    var_6 = 4571
    var_7 = module_0.Position(var_6, var_5, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 4571
    assert var_7.column_no is False
    assert var_7.char_index == 4571
    var_8 = var_7.__eq__(var_6)
    assert var_8 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_4, messages=var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = False
    var_5 = True
    var_6 = module_0.Position(var_4, var_1, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is False
    assert var_6.column_no is None
    assert var_6.char_index is True
    var_7 = var_6.__eq__(var_1)
    assert var_7 is False
    module_0.BaseError(position=var_1, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = '%'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.messages()
    var_5 = var_4.__repr__()
    assert var_5 == "[Message(text='%', code='custom')]"
    var_6 = var_3.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = var_6.__repr__()
    assert var_7 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_8 = var_3.messages(add_prefix=var_4)
    var_9 = var_3.__str__()
    assert var_9 == '%'
    module_0.ParseError(key=var_2, position=var_0)

def test_case_19():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {'name': 'test'}
    assert var_3.error is None
    var_4 = list(var_3)
    var_5 = len(var_4)
    assert var_5 == 2
    var_6 = 'error text'
    var_7 = 'err_code'
    var_8 = module_0.Message(text=var_6, code=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'error text'
    assert var_8.code == 'err_code'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error == 2
    var_10 = list(var_9)
    var_11 = len(var_10)
    assert var_11 == 2
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = list(var_12)
    var_14 = len(var_13)
    assert var_14 == 2

def test_case_20():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_3.__repr__()
    assert var_4 == "[Message(text='%', code='custom')]"
    var_5 = 640
    var_6 = False
    var_7 = 4571
    var_8 = module_0.Position(var_5, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == 640
    assert var_8.column_no is False
    assert var_8.char_index == 4571
    var_9 = var_8.__eq__(var_5)
    assert var_9 is False
    var_10 = var_2.__hash__()
    assert var_10 == 2305123341543567570
    var_11 = module_0.BaseError(text=var_4, position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, position=var_10, messages=var_10)

def test_case_21():
    var_0 = None
    var_1 = '%'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.messages()
    var_5 = var_4.__repr__()
    assert var_5 == "[Message(text='%', code='custom')]"
    var_6 = var_3.get(var_2, var_5)
    assert var_6 == "[Message(text='%', code='custom')]"
    var_7 = var_6.__repr__()
    assert var_7 == '"[Message(text=\'%\', code=\'custom\')]"'
    var_8 = module_0.Position(var_0, var_2, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert var_8.char_index == "[Message(text='%', code='custom')]"
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.BaseError(text=var_6, key=var_9, position=var_6, messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_6.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = '%'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.messages()
    var_5 = var_4.__repr__()
    assert var_5 == "[Message(text='%', code='custom')]"
    var_6 = var_3.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = var_6.__repr__()
    assert var_7 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_8 = var_6.__eq__(var_0)
    var_9 = var_3.messages()
    var_10 = 4571
    var_11 = False
    var_12 = module_0.Position(var_0, var_11, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is None
    assert var_12.column_no is False
    assert var_12.char_index == 4571
    module_0.ValidationError(code=var_2, key=var_6, messages=var_6)

def test_case_23():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_3.__repr__()
    assert var_4 == "[Message(text='%', code='custom')]"
    var_5 = -258
    var_6 = False
    var_7 = 4571
    var_8 = module_0.Position(var_5, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == -258
    assert var_8.column_no is False
    assert var_8.char_index == 4571
    var_9 = var_8.__eq__(var_5)
    assert var_9 is False
    var_10 = var_2.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_10, messages=var_3)

def test_case_24():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_2.messages()
    var_5 = var_2.__len__()
    assert var_5 == 1
    var_6 = var_5.__repr__()
    assert var_6 == '1'
    var_7 = '\n`<?Rt'
    var_8 = module_0.Message(text=var_7, code=var_1, index=var_5, end_position=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '\n`<?Rt'
    assert var_8.code == 'custom'
    assert var_8.index == 1
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_2.get(var_1)
    var_10 = var_8.__repr__()
    assert var_10 == "Message(text='\\n`<?Rt', code='custom', index=1)"
    var_11 = False
    var_12 = 3226
    var_13 = module_0.Position(var_12, var_11, var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 3226
    assert var_13.column_no is False
    assert f'{type(var_13.char_index).__module__}.{type(var_13.char_index).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.char_index) == 1
    var_14 = var_9.__eq__(var_5)
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_5)

def test_case_25():
    var_0 = None
    var_1 = '%'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "ValuesView(ValidationError(text='%', code='custom'))"
    var_6 = var_4.__repr__()
    assert var_6 == "ValuesView(ValidationError(text='%', code='custom'))"
    var_7 = True
    var_8 = module_0.Position(var_4, var_2, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.line_no).__module__}.{type(var_8.line_no).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8.line_no) == 1
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = module_0.Position(var_4, var_6, var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.line_no).__module__}.{type(var_9.line_no).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_9.line_no) == 1
    assert var_9.column_no == "ValuesView(ValidationError(text='%', code='custom'))"
    assert var_9.char_index is True
    var_10 = var_9.__eq__(var_0)
    assert var_10 is False
    var_11 = var_3.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_11) == 1
    var_12 = module_0.BaseError(text=var_4, key=var_2, position=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    var_13 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_13, code=var_6, messages=var_11)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = ''
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ''
    assert var_1.code == ''
    assert var_1.index == ['']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = '7dxHq7]![\n@'
    var_4 = None
    var_5 = module_0.Message(text=var_3, key=var_3, position=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '7dxHq7]![\n@'
    assert var_5.code == 'custom'
    assert var_5.index == ['7dxHq7]![\n@']
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_2)
    assert var_6 is False
    module_0.ValidationError(text=var_2, code=var_2, messages=var_2)

def test_case_27():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_3.__repr__()
    assert var_4 == "[Message(text='%', code='custom')]"
    var_5 = var_2.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_7 = var_5.__eq__(var_0)
    var_8 = var_2.messages()
    var_9 = var_5.__repr__()
    assert var_9 == "ItemsView(ValidationError(text='%', code='custom'))"
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_3, error=var_5)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = None
    var_3 = '%'
    var_4 = module_0.ValidationError(text=var_3, code=var_0, messages=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.messages()
    var_6 = var_5.__repr__()
    assert var_6 == "[Message(text='%', code='custom')]"
    var_7 = True
    var_8 = var_4.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = var_8.__repr__()
    assert var_9 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_10 = var_8.__eq__(var_2)
    var_11 = var_4.messages()
    var_12 = var_8.__repr__()
    assert var_12 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_13 = module_0.ValidationResult(value=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value == "ItemsView(ValidationError(text='%', code='custom'))"
    assert var_13.error is None
    var_14 = var_8.__repr__()
    assert var_14 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_15 = var_8.__str__()
    assert var_15 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_16 = module_0.ParseError(code=var_2, messages=var_11)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_8.__len__()
    assert var_17 == 1
    var_18 = var_17.__repr__()
    assert var_18 == '1'
    var_19 = var_16.__str__()
    assert var_19 == '%'
    var_20 = module_0.Message(text=var_3, key=var_7, position=var_4, start_position=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == '%'
    assert var_20.code == 'custom'
    assert var_20.index == [True]
    assert f'{type(var_20.start_position).__module__}.{type(var_20.start_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20.start_position) == 1
    assert f'{type(var_20.end_position).__module__}.{type(var_20.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20.end_position) == 1
    var_21 = var_20.__repr__()
    assert var_21 == "Message(text='%', code='custom', index=[True], position=ValidationError(text='%', code='custom'))"
    module_0.ValidationError(code=var_2, position=var_2, messages=var_6)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.messages()
    var_6 = var_3.__repr__()
    assert var_6 == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    var_7 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is None
    assert var_7.char_index is None
    var_8 = module_0.Position(var_4, var_2, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is False
    assert var_8.column_no is None
    assert var_8.char_index is False
    var_9 = var_7.__eq__(var_2)
    assert var_9 is False
    var_10 = module_0.ParseError(text=var_1, key=var_9, messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.__eq__(var_3)
    assert var_11 is False
    var_12 = var_10.__str__()
    assert var_12 == "{False: 'ValidationResult(value=None)'}"
    var_13 = '\r<|<-=r\tZ&A(|PEkr\x0b'
    var_14 = var_13.__hash__()
    assert var_14 == -7910841165649951711
    var_15 = var_10.values()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_15) == 1
    var_16 = var_15.__eq__(var_15)
    assert var_16 is True
    var_16.__contains__(var_2)

def test_case_30():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = None
    var_3 = '%'
    var_4 = None
    var_5 = module_0.ValidationError(text=var_3, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_6.__repr__()
    assert var_7 == "[Message(text='%', code='custom')]"
    var_8 = var_5.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = var_8.__repr__()
    assert var_9 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_10 = var_8.__eq__(var_2)
    var_11 = var_5.messages()
    var_12 = var_8.__repr__()
    assert var_12 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_13 = module_0.ValidationResult(value=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value == "ItemsView(ValidationError(text='%', code='custom'))"
    assert var_13.error is None
    var_14 = var_8.__repr__()
    assert var_14 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_15 = var_5.messages()
    var_16 = var_8.__str__()
    assert var_16 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_17 = module_0.ParseError(code=var_2, messages=var_15)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_17) == 1
    var_18 = var_8.__len__()
    assert var_18 == 1
    var_19 = var_18.__repr__()
    assert var_19 == '1'
    var_20 = var_17.__str__()
    assert var_20 == '%'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_14, key=var_17, index=var_8, position=var_2)

def test_case_31():
    var_0 = None
    var_1 = '%'
    var_2 = module_0.ValidationError(text=var_1, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_3.__repr__()
    assert var_4 == "[Message(text='%', code='custom')]"
    var_5 = var_2.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_5) == 1
    var_6 = var_2.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = var_6.__repr__()
    assert var_7 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_8 = var_6.__eq__(var_0)
    var_9 = var_2.messages()
    var_10 = var_6.__repr__()
    assert var_10 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_11 = module_0.ValidationResult(value=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value == "ItemsView(ValidationError(text='%', code='custom'))"
    assert var_11.error is None
    var_12 = var_6.__repr__()
    assert var_12 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_13 = var_2.messages()
    var_14 = var_6.__str__()
    assert var_14 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_15 = module_0.ParseError(code=var_0, messages=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = module_0.BaseError(text=var_5, position=var_6)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_16) == 1
    var_17 = var_16.__str__()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, index=var_6, position=var_6, start_position=var_5, end_position=var_5)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = None
    var_3 = '%'
    var_4 = None
    var_5 = module_0.ValidationError(text=var_3, code=var_0, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_6.__repr__()
    assert var_7 == "[Message(text='%', code='custom')]"
    var_8 = var_5.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = var_8.__repr__()
    assert var_9 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_10 = var_8.__eq__(var_2)
    var_11 = var_5.messages()
    var_12 = var_8.__repr__()
    assert var_12 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_13 = module_0.ValidationResult(value=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value == "ItemsView(ValidationError(text='%', code='custom'))"
    assert var_13.error is None
    var_14 = var_8.__repr__()
    assert var_14 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_15 = var_5.messages()
    var_16 = var_8.__str__()
    assert var_16 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_17 = var_5.__str__()
    assert var_17 == '%'
    var_18 = var_8.__eq__(var_0)
    var_19 = module_0.ParseError(text=var_8, code=var_0, position=var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_19) == 1
    var_20 = var_8.__len__()
    assert var_20 == 1
    var_21 = var_20.__repr__()
    assert var_21 == '1'
    var_22 = var_8.__str__()
    assert var_22 == "ItemsView(ValidationError(text='%', code='custom'))"
    var_23 = module_0.Message(text=var_3, key=var_18, start_position=var_8)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == '%'
    assert var_23.code == 'custom'
    assert f'{type(var_23.index).__module__}.{type(var_23.index).__qualname__}' == 'builtins.list'
    assert len(var_23.index) == 1
    assert f'{type(var_23.start_position).__module__}.{type(var_23.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_23.start_position) == 1
    assert var_23.end_position is None
    var_24 = var_23.__repr__()
    assert var_24 == "Message(text='%', code='custom', index=[NotImplemented], start_position=ItemsView(ValidationError(text='%', code='custom')), end_position=None)"
    module_0.ValidationError(position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.messages()
    var_6 = module_0.Message(text=var_1, code=var_1, key=var_1, end_position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'ValidationResult(value=None)'
    assert var_6.index == ['ValidationResult(value=None)']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__repr__()
    assert var_7 == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    var_8 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert var_8.char_index is None
    var_9 = module_0.Position(var_4, var_2, var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is False
    assert var_9.column_no is None
    assert var_9.char_index is False
    var_10 = module_0.ParseError(text=var_1, key=var_4, messages=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_5.__repr__()
    assert var_11 == "[Message(text='ValidationResult(value=None)', code='custom')]"
    var_12 = 'ZP3 E>F2\\>C,hh'
    var_13 = var_10.__str__()
    assert var_13 == "{False: 'ValidationResult(value=None)'}"
    var_14 = var_6.__eq__(var_6)
    assert var_14 is True
    var_15 = var_0.__bool__()
    assert var_15 is True
    var_16 = var_10.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    module_0.ParseError(text=var_12, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = None
    var_3 = module_0.ValidationError(text=var_0, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.messages()
    var_6 = var_3.__repr__()
    assert var_6 == "ValidationError(text=ValidationResult(value=None), code='custom')"
    var_7 = module_0.Position(var_6, var_2, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == "ValidationError(text=ValidationResult(value=None), code='custom')"
    assert var_7.column_no is None
    assert var_7.char_index == "ValidationError(text=ValidationResult(value=None), code='custom')"
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = var_3.__str__()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    module_0.ParseError(code=var_9, position=var_7)

def test_case_35():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = var_0.__bool__()
    assert var_2 is True
    var_3 = 'KN#G7Y1Z>3<+I:r'
    var_4 = None
    var_5 = module_0.ValidationError(text=var_3, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.__eq__(var_0)
    assert var_6 is False
    var_7 = var_5.messages()
    var_8 = var_7.__repr__()
    assert var_8 == "[Message(text='KN#G7Y1Z>3<+I:r', code='custom')]"
    var_9 = var_5.__repr__()
    assert var_9 == "ValidationError(text='KN#G7Y1Z>3<+I:r', code='custom')"
    var_10 = 660
    var_11 = module_0.Position(var_4, var_10, var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert var_11.column_no == 660
    assert var_11.char_index is None
    var_12 = module_0.Position(var_6, var_4, var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is False
    assert var_12.column_no is None
    assert var_12.char_index is False
    var_13 = var_11.__eq__(var_4)
    assert var_13 is False
    var_14 = var_5.__str__()
    assert var_14 == 'KN#G7Y1Z>3<+I:r'
    var_15 = module_0.ParseError(text=var_9, position=var_12)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_5.__iter__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_16, code=var_4, index=var_4, position=var_16, end_position=var_16)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = 'mNP*1X-'
    var_3 = None
    var_4 = module_0.ValidationError(text=var_1, key=var_2, messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__len__()
    assert var_5 == 1
    var_6 = var_0.__bool__()
    assert var_6 is True
    var_7 = var_4.messages(add_prefix=var_2)
    var_8 = var_4.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = var_8.__repr__()
    assert var_9 == "ItemsView(ValidationError([Message(text='ValidationResult(value=None)', code='custom', index=['mNP*1X-'])]))"
    var_8.__getitem__(var_8)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = '%'
    var_3 = None
    var_4 = module_0.ValidationError(text=var_2, messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_4.messages()
    var_7 = var_6.__repr__()
    assert var_7 == "[Message(text='%', code='custom')]"
    var_8 = 640
    var_9 = module_0.Position(var_3, var_8, var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no == 640
    assert var_9.char_index is None
    var_10 = module_0.Position(var_5, var_3, var_5)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is False
    assert var_10.column_no is None
    assert var_10.char_index is False
    var_11 = var_9.__eq__(var_10)
    assert var_11 is False
    var_12 = var_4.__str__()
    assert var_12 == '%'
    var_13 = var_4.__str__()
    assert var_13 == '%'
    var_14 = var_4.get(var_3)
    var_15 = var_4.__eq__(var_6)
    assert var_15 is False
    var_16 = var_4.__eq__(var_15)
    assert var_16 is False
    var_17 = var_16.__hash__()
    assert var_17 == 0
    var_18 = module_0.Message(text=var_7, end_position=var_14)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text == "[Message(text='%', code='custom')]"
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert var_18.start_position is None
    assert var_18.end_position is None
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = 'mNP*1X-'
    var_3 = None
    var_4 = module_0.ValidationError(text=var_1, key=var_2, messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__len__()
    assert var_5 == 1
    var_6 = var_0.__bool__()
    assert var_6 is True
    var_7 = 'KN#G7Y1Z>3<+I:r'
    var_8 = None
    var_9 = module_0.ValidationError(text=var_7, messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_9.__eq__(var_0)
    assert var_10 is False
    var_11 = var_9.messages()
    var_12 = 1860
    var_13 = module_0.Position(var_3, var_12, var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no == 1860
    assert var_13.char_index is None
    var_14 = True
    var_15 = True
    var_16 = module_0.Position(var_3, var_15, var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no is None
    assert var_16.column_no is True
    assert var_16.char_index is True
    var_17 = var_13.__eq__(var_16)
    assert var_17 is False
    var_18 = var_4.__str__()
    assert var_18 == "{'mNP*1X-': 'ValidationResult(value=None)'}"
    module_0.ParseError(text=var_3, key=var_8)

def test_case_39():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    var_5 = var_3.messages()
    var_6 = var_5.__repr__()
    assert var_6 == "[Message(text='ValidationResult(value=None)', code='custom')]"
    var_7 = module_0.Message(text=var_1, code=var_1, key=var_6, end_position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'ValidationResult(value=None)'
    assert var_7.code == 'ValidationResult(value=None)'
    assert var_7.index == ["[Message(text='ValidationResult(value=None)', code='custom')]"]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__repr__()
    assert var_8 == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    var_9 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no is None
    assert var_9.char_index is None
    var_10 = module_0.Position(var_4, var_2, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_10.line_no).__module__}.{type(var_10.line_no).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_10.line_no) == 1
    assert var_10.column_no is None
    assert f'{type(var_10.char_index).__module__}.{type(var_10.char_index).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_10.char_index) == 1
    var_11 = var_9.__eq__(var_2)
    assert var_11 is False
    var_12 = module_0.ParseError(text=var_1, key=var_11, messages=var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_12.messages()
    var_14 = var_13.__repr__()
    assert var_14 == "[Message(text='ValidationResult(value=None)', code='custom', index=[False])]"
    var_15 = var_12.__str__()
    assert var_15 == "{False: 'ValidationResult(value=None)'}"
    var_16 = var_3.__str__()
    assert var_16 == 'ValidationResult(value=None)'
    var_17 = var_4.__eq__(var_2)
    var_18 = var_12.__hash__()
    assert var_18 == 7358965412909632342
    var_19 = module_0.ValidationResult(error=var_3)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_19.value is None
    assert f'{type(var_19.error).__module__}.{type(var_19.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19.error) == 1
    var_20 = var_19.__repr__()
    assert var_20 == "ValidationResult(error=ValidationError(text='ValidationResult(value=None)', code='custom'))"
    var_4.values()

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.messages()
    var_6 = var_5.__repr__()
    assert var_6 == "[Message(text='ValidationResult(value=None)', code='custom')]"
    var_7 = module_0.Message(text=var_1, code=var_1, key=var_6, end_position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'ValidationResult(value=None)'
    assert var_7.code == 'ValidationResult(value=None)'
    assert var_7.index == ["[Message(text='ValidationResult(value=None)', code='custom')]"]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_3.__repr__()
    assert var_8 == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    var_9 = var_7.__repr__()
    assert var_9 == 'Message(text=\'ValidationResult(value=None)\', code=\'ValidationResult(value=None)\', index=["[Message(text=\'ValidationResult(value=None)\', code=\'custom\')]"])'
    var_10 = module_0.Position(var_8, var_4, var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    assert var_10.column_no is False
    assert f'{type(var_10.char_index).__module__}.{type(var_10.char_index).__qualname__}' == 'typesystem.base.Message'
    var_11 = 2351
    var_12 = module_0.Position(var_11, var_4, var_2)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == 2351
    assert var_12.column_no is False
    assert var_12.char_index is None
    var_13 = var_10.__eq__(var_2)
    assert var_13 is False
    var_14 = [var_7, var_7, var_7, var_7]
    var_15 = module_0.ParseError(code=var_2, messages=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_15.messages()
    var_17 = var_0.__repr__()
    assert var_17 == 'ValidationResult(value=None)'
    var_18 = var_15.__str__()
    assert var_18 == '{"[Message(text=\'ValidationResult(value=None)\', code=\'custom\')]": \'ValidationResult(value=None)\'}'
    var_19 = var_7.__eq__(var_2)
    assert var_19 is False
    var_20 = var_15.__hash__()
    assert var_20 == 8492881045419273136
    var_21 = var_15.values()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_21) == 1
    var_22 = var_0.__bool__()
    assert var_22 is True
    var_21.__bool__()

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'error_code'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'error_code'
    assert var_1.code == 'error_code'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ValidationError(text=var_0, code=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = str(var_2)
    var_4 = 'Field error'
    var_5 = 'username'
    var_6 = module_0.Message(text=var_4, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Field error'
    assert var_6.code == 'custom'
    assert var_6.index == ['username']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = [var_6]
    var_8 = module_0.ValidationError(messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = str(var_8)
    assert var_9 == "{'username': 'Field error'}"
    var_10 = 'Nested error'
    var_11 = 0
    var_12 = 'email'
    var_13 = var_1.__eq__(var_11)
    assert var_13 is False
    var_14 = [var_10, var_11, var_12]
    var_15 = module_0.Message(text=var_10, index=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'Nested error'
    assert var_15.code == 'custom'
    assert var_15.index == ['Nested error', 0, 'email']
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = 'root'
    var_17 = module_0.Message(text=var_3, key=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'error_code'
    assert var_17.code == 'custom'
    assert var_17.index == ['root']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = [var_15, var_17]
    var_19 = module_0.ValidationError(messages=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19) == 2
    var_20 = str(var_19)
    var_21 = var_2.values()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_21) == 1
    var_3.__getitem__(var_20)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = None
    var_4 = module_0.ValidationError(text=var_1, code=var_2, messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_5) == 1
    var_6 = var_4.messages()
    var_7 = var_5.__hash__()
    var_8 = module_0.Message(text=var_4, code=var_1, start_position=var_2, end_position=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_8.text).__module__}.{type(var_8.text).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8.text) == 1
    assert var_8.code == 'ValidationResult(value=None)'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_5.__repr__()
    assert var_9 == "ValuesView(ValidationError(text='ValidationResult(value=None)', code='custom'))"
    var_10 = module_0.Position(var_2, var_3, var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is None
    assert var_10.char_index is None
    var_11 = 199
    var_12 = module_0.Position(var_11, var_2, var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == 199
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = b'\x9a\xe5Q\x9e\x02\x93\xd7*\xdc\xf5q\xfdgy\xfc\xc1\xc7to\x9c'
    var_14 = var_5.__eq__(var_13)
    var_15 = [var_8, var_8]
    var_16 = module_0.ParseError(text=var_3, code=var_2, messages=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_5.__repr__()
    assert var_17 == "ValuesView(ValidationError(text='ValidationResult(value=None)', code='custom'))"
    var_18 = var_16.__str__()
    assert var_18 == "{'': ValidationError(text='ValidationResult(value=None)', code='custom')}"
    var_19 = '3\r<|<-=r\tZ&A(|PEkr\x0b'
    var_20 = var_8.__eq__(var_2)
    assert var_20 is False
    var_21 = var_5.__hash__()
    var_22 = var_0.__bool__()
    assert var_22 is True
    var_23 = var_16.values()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_23) == 1
    var_24 = var_23.__repr__()
    assert var_24 == "ValuesView(ParseError([Message(text=ValidationError(text='ValidationResult(value=None)', code='custom'), code='ValidationResult(value=None)'), Message(text=ValidationError(text='ValidationResult(value=None)', code='custom'), code='ValidationResult(value=None)')]))"
    var_25 = module_0.ParseError(text=var_1, key=var_19, position=var_23)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_25) == 1
    var_26 = var_25.__iter__()
    var_27 = var_4.__eq__(var_22)
    assert var_27 is False
    var_27.__len__()

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.messages()
    var_6 = module_0.Message(text=var_1, code=var_1, key=var_1, end_position=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'ValidationResult(value=None)'
    assert var_6.index == ['ValidationResult(value=None)']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3.__repr__()
    assert var_7 == "ValidationError(text='ValidationResult(value=None)', code='custom')"
    var_8 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert var_8.char_index is None
    var_9 = module_0.Position(var_4, var_2, var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is False
    assert var_9.column_no is None
    assert var_9.char_index is False
    var_10 = var_8.__eq__(var_2)
    assert var_10 is False
    var_11 = module_0.ParseError(text=var_1, key=var_10, messages=var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_5.__repr__()
    assert var_12 == "[Message(text='ValidationResult(value=None)', code='custom')]"
    var_13 = var_11.__str__()
    assert var_13 == "{False: 'ValidationResult(value=None)'}"
    var_14 = var_6.__eq__(var_6)
    assert var_14 is True
    var_15 = var_3.__hash__()
    assert var_15 == 2305123341543567570
    var_16 = var_0.__bool__()
    assert var_16 is True
    var_17 = var_0.__bool__()
    assert var_17 is True
    var_18 = var_0.__bool__()
    assert var_18 is True
    var_19 = var_11.keys()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_19) == 1
    var_20 = module_0.ParseError(text=var_13, code=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_20) == 1
    var_21 = var_20.__eq__(var_3)
    assert var_21 is False
    var_22 = var_3.__len__()
    assert var_22 == 1
    var_23 = var_20.keys()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_23) == 1
    var_24 = var_23.__contains__(var_22)
    assert var_24 is False
    var_23.items()