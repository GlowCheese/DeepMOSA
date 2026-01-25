# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 2
    var_4 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 2
    assert var_4.char_index == 1
    var_5 = None
    var_6 = var_2.__eq__(var_5)
    assert var_6 is False
    var_7 = 'Error'
    var_8 = module_0.Message(text=var_7, start_position=var_2, end_position=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = module_0.Message(text=var_7, start_position=var_2, end_position=var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error'
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.end_position).__module__}.{type(var_9.end_position).__qualname__}' == 'typesystem.base.Position'
    var_10 = var_8 == var_9
    assert var_10 is False

def test_case_1():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.ValidationError(text=var_0, code=var_0, key=var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = None
    var_4 = module_0.ValidationError(code=var_3, messages=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_4.__bool__()

def test_case_4():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()

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
    var_1 = False
    var_2 = module_0.Position(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = module_0.ValidationError(text=var_2, key=var_0, position=var_0, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.messages()
    var_5 = var_3.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "ItemsView(ValidationError(text=Position(line_no=False, column_no=None, char_index=None), code='custom'))"

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = '`]'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_2) == 1
    var_3 = None
    module_0.ValidationError(code=var_3, messages=var_2)

def test_case_9():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'

def test_case_10():
    var_0 = 'h 2\nY `3o0JNw6b'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, messages=var_0)

def test_case_11():
    var_0 = None
    var_1 = None
    var_2 = 'Q[/J7F)bh\\i'
    var_3 = module_0.Message(text=var_1, code=var_2, key=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'Q[/J7F)bh\\i'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None

def test_case_12():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationError(code=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_3.__len__()
    assert var_6 == 1
    var_7 = var_1.__hash__()
    assert var_7 == 696657079373750063
    var_8 = module_0.ValidationResult(value=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == 696657079373750063
    assert var_8.error is None
    var_9 = var_8.__repr__()
    assert var_9 == 'ValidationResult(value=696657079373750063)'

def test_case_13():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = '}kfR'
    var_2 = False
    var_3 = None
    var_4 = module_0.Position(var_2, var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert var_4.column_no is None
    assert var_4.char_index is False
    var_5 = module_0.Message(text=var_1, index=var_0, position=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '}kfR'
    assert var_5.code == 'custom'
    assert f'{type(var_5.index).__module__}.{type(var_5.index).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_5.__hash__()
    assert var_6 == -6466911654036342760
    var_7 = var_5.__repr__()
    assert var_7 == "Message(text='}kfR', code='custom', index=ValidationResult(value=None), position=Position(line_no=False, column_no=None, char_index=False))"

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '&Qrj3R3D!DMAg^'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1
    var_4 = None
    module_0.ValidationError(position=var_4)

def test_case_15():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_2.__repr__()
    assert var_3 == 'False'

def test_case_16():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__repr__()
    assert var_3 == "ValidationError(text='dO', code='custom')"

def test_case_17():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.value) == 1
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == "ValidationResult(value=ValidationError(text='dO', code='custom'))"

def test_case_18():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = var_3.__repr__()
    assert var_4 == "ValidationResult(error=ValidationError(text='dO', code='custom'))"

def test_case_19():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True

def test_case_20():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '`e'

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_1, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = module_0.ValidationError(text=var_2, key=var_0, position=var_0, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    var_5 = [var_4, var_4, var_4, var_4]
    module_0.ValidationError(text=var_0, key=var_0, position=var_2, messages=var_5)

def test_case_22():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationError(code=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_6.value).__module__}.{type(var_6.value).__qualname__}' == 'builtins.list'
    assert len(var_6.value) == 1
    assert var_6.error is None
    var_7 = var_6.__repr__()
    assert var_7 == "ValidationResult(value=[Message(text='`e', code='custom')])"

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages(add_prefix=var_0)
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    module_0.ValidationError(code=var_3, messages=var_4)

def test_case_24():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = False
    var_3 = None
    var_4 = module_0.Position(var_2, var_1, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is False
    assert f'{type(var_4.column_no).__module__}.{type(var_4.column_no).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.column_no) == 1
    assert var_4.char_index is None
    var_5 = module_0.ValidationError(text=var_0, position=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.__len__()
    assert var_6 == 1
    var_7 = var_6.__repr__()
    assert var_7 == '1'

def test_case_25():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = 'prl"`v:\\3\']J'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, key=var_3, index=var_3, end_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_2) == 1
    module_0.ValidationError(code=var_2, messages=var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationError(code=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = "}'fR"
    var_8 = module_0.Message(text=var_0, code=var_4, key=var_7, position=var_3, start_position=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == '`e'
    assert var_8.code == 'custom'
    assert var_8.index == ["}'fR"]
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8.start_position) == 1
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8.end_position) == 1
    var_9 = var_1.__len__()
    assert var_9 == 1
    var_10 = var_5.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    var_11 = var_8.__repr__()
    assert var_11 == 'Message(text=\'`e\', code=\'custom\', index=["}\'fR"], position=ItemsView(ValidationError(text=\'`e\', code=\'custom\')))'
    var_12 = 717
    module_0.ParseError(key=var_12, messages=var_2)

def test_case_28():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationError(code=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_3, index=var_4, position=var_3, start_position=var_3)

def test_case_29():
    var_0 = 'dO'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = None
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "KeysView(ValidationError(text='dO', code='custom'))"
    var_6 = -2932
    var_7 = module_0.Position(var_6, var_3, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == -2932
    assert var_7.column_no is None
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7.char_index) == 1
    var_8 = var_1.messages()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_7, index=var_3, position=var_4, start_position=var_3, end_position=var_7)

def test_case_30():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationError(code=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = module_0.Message(text=var_3, key=var_4, start_position=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_7.text).__module__}.{type(var_7.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_7.text) == 1
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_7.start_position) == 1
    assert var_7.end_position is None
    var_8 = var_3.__len__()
    assert var_8 == 1
    var_9 = var_7.__repr__()
    assert var_9 == "Message(text=ItemsView(ValidationError(text='`e', code='custom')), code='custom', start_position=ItemsView(ValidationError(text='`e', code='custom')), end_position=None)"
    var_10 = var_5.messages()

def test_case_31():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_3.__hash__()
    assert var_4 == 0
    var_5 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.text) == 1
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.start_position) == 1
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.end_position) == 1
    var_6 = var_1.__str__()
    assert var_6 == '`e'
    var_7 = var_6.__repr__()
    assert var_7 == "'`e'"
    var_8 = var_5.__repr__()
    assert var_8 == "Message(text=ValidationError(text='`e', code='custom'), code='custom', position=ValidationError(text='`e', code='custom'))"

def test_case_32():
    var_0 = None
    var_1 = 'u>'
    var_2 = module_0.Message(text=var_0, code=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'u>'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 2699970444524581848
    var_4 = var_2.__eq__(var_0)
    assert var_4 is False
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text=None, code='u>')"

def test_case_33():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationError(code=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_3, error=var_5)

def test_case_34():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'max_length'
    var_4 = module_0.Message(text=var_0, code=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'max_length'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2 == var_4
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = None
    var_4 = module_0.ValidationError(code=var_3, messages=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_6.text).__module__}.{type(var_6.text).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.text) == 1
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.start_position) == 1
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.end_position) == 1
    var_7 = module_0.Message(text=var_0, index=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '`e'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = "B,2q&i'y.>hj3&V a[w"
    var_9 = module_0.Message(text=var_8, position=var_3, start_position=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == "B,2q&i'y.>hj3&V a[w"
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_6.__hash__()
    assert var_10 == 2474166882269941120
    var_11 = var_7.__eq__(var_6)
    assert var_11 is False
    var_12 = var_4.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_1.__contains__(var_12)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = []
    module_0.ParseError(messages=var_0)

def test_case_37():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 2
    var_4 = module_0.Position(var_0, var_3, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 2
    assert var_4.char_index == 1
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

def test_case_38():
    var_0 = 'Nested error'
    var_1 = 'nested'
    var_2 = 'child'
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Nested error'
    assert var_4.code == 'nested'
    assert var_4.index == ['nested', 'child']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = str(var_6)
    var_8 = {var_2: var_0}
    var_9 = {var_2: var_8}
    var_10 = str(var_9)

def test_case_39():
    var_0 = '`e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.__str__()
    assert var_3 == '`e'
    var_4 = module_0.ValidationError(text=var_0, key=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = None
    var_6 = var_4.messages(add_prefix=var_5)
    var_7 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = None
    var_9 = var_1.__eq__(var_8)
    assert var_9 is False
    var_10 = False
    var_11 = None
    var_12 = var_4.__repr__()
    assert var_12 == "ValidationError([Message(text='`e', code='custom', index=['`e'])])"
    var_13 = module_0.Position(var_11, var_10, var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no is False
    assert var_13.char_index is False
    var_14 = "VHr8c b'L+6kRJ8\nS"
    var_15 = module_0.Message(text=var_14, position=var_8, end_position=var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == "VHr8c b'L+6kRJ8\nS"
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_15.__hash__()
    assert var_16 == 2474166882269941120
    var_17 = var_4.__eq__(var_5)
    assert var_17 is False
    var_18 = var_17.__hash__()
    assert var_18 == 0
    var_19 = var_15.__eq__(var_8)
    assert var_19 is False
    var_20 = var_17.__str__()
    assert var_20 == 'False'
    var_21 = var_15.__repr__()
    assert var_21 == 'Message(text="VHr8c b\'L+6kRJ8\\nS", code=\'custom\')'
    var_22 = var_17.__repr__()
    assert var_22 == 'False'

def test_case_40():
    var_0 = 'Error'
    var_1 = 'users'
    var_2 = 0
    var_3 = [var_1, var_2]
    var_4 = module_0.Message(text=var_0, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'custom'
    assert var_4.index == ['users', 0]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 1
    var_6 = [var_1, var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == ['users', 1]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_4 == var_7
    assert var_8 is False

def test_case_41():
    var_0 = 1
    var_1 = 0
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1
    assert var_2.column_no == 1
    assert var_2.char_index == 0
    var_3 = 2
    var_4 = 10
    var_5 = module_0.Position(var_3, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 2
    assert var_5.column_no == 1
    assert var_5.char_index == 10
    var_6 = 'Error'
    var_7 = module_0.Message(text=var_6, start_position=var_2, end_position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = module_0.Message(text=var_6, start_position=var_5, end_position=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = var_7 == var_8
    assert var_9 is False

def test_case_42():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = [var_1]
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 1'
    assert var_3.code == 'code1'
    assert var_3.index == ['code1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'code2'
    var_5 = 'field2'
    var_6 = [var_3, var_3]
    var_7 = module_0.BaseError(messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = str(var_7)
    var_9 = {var_4: var_0, var_5: var_4}
    var_10 = str(var_9)

def test_case_43():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1, index=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error 1'
    assert var_2.code == 'code1'
    assert var_2.index == 'Error 1'
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = [var_2, var_2]
    var_4 = module_0.BaseError(messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = repr(var_4)