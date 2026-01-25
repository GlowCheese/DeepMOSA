# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = module_0.ValidationError(text=var_2, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1

def test_case_1():
    var_0 = None
    var_1 = None
    var_2 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None

@pytest.mark.xfail(strict=True)
def test_case_2():
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    module_0.ValidationError(key=var_0, position=var_0, messages=var_1)

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_5():
    var_0 = 410.5287
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == pytest.approx(410.5287, abs=0.01, rel=0.01)
    assert var_1.error is None

def test_case_6():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = var_1.__bool__()
    assert var_3 is True

def test_case_7():
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

def test_case_8():
    var_0 = None
    var_1 = 'I%wrXC]%<O-d'
    var_2 = module_0.ValidationError(text=var_1, position=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ',P9akHtEE@B{y:?OyU\x0c'
    var_1 = None
    module_0.ValidationError(text=var_0, code=var_1, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = 'EEua3P{s|(]\x0cy\rA%Sy.'
    var_2 = module_0.BaseError(text=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_4 = var_2.__eq__(var_0)
    assert var_4 is False
    var_2.__getitem__(var_0)

def test_case_11():
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
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='custom')"

def test_case_12():
    var_0 = '\x0b8P\x0b\x0cl8"^X'
    var_1 = None
    var_2 = module_0.BaseError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = var_3.__repr__()
    assert var_4 == 'ItemsView(BaseError(text=\'\\x0b8P\\x0b\\x0cl8"^X\', code=\'custom\'))'

def test_case_13():
    var_0 = None
    var_1 = 'IWwrXC]%<O-d'
    var_2 = module_0.ValidationError(text=var_1, position=var_1, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_0)
    with pytest.raises(AssertionError):
        module_0.BaseError()

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '$TMa'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = False
    var_3 = var_1.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_2)
    var_5 = var_1.__iter__()
    var_6 = var_1.__repr__()
    assert var_6 == "BaseError(text='$TMa', code='custom')"
    var_5.values()

def test_case_15():
    var_0 = 'custom'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'custom'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'custom'
    assert var_3.code == 'max_length'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1 == var_3
    assert var_4 is False

def test_case_16():
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
    var_9 = var_6.__repr__()
    assert var_9 == "Message(text='Error', code='custom', position=Position(line_no=1, column_no=1, char_index=0))"

def test_case_17():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = None
    var_4 = -1511
    var_5 = module_0.Position(var_2, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is True
    assert var_5.column_no is True
    assert var_5.char_index == -1511
    var_6 = module_0.Position(var_0, var_4, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is None
    assert var_6.column_no == -1511
    assert var_6.char_index is True
    var_7 = var_6.__eq__(var_3)
    assert var_7 is False
    var_8 = var_1.__hash__()
    assert var_8 == 5748167320160572150
    var_9 = var_5.__eq__(var_0)
    assert var_9 is False
    var_10 = var_5.__eq__(var_5)
    assert var_10 is True

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = 'IWwrX]%<O-d'
    var_2 = module_0.ValidationError(text=var_1, position=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ValidationError(text='IWwrX]%<O-d', code='custom')"
    var_4 = var_2.messages()
    var_3.__getitem__(var_0)

def test_case_19():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = None
    var_4 = False
    var_5 = module_0.Position(var_4, var_0, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no is None
    assert var_5.char_index is None
    var_6 = var_5.__eq__(var_3)
    assert var_6 is False

def test_case_20():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_1.__bool__()
    assert var_5 is True
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = None
    var_8 = -1511
    var_9 = var_4.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_4.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = module_0.BaseError(text=var_10, position=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_11.__str__()
    assert var_12 == "Message(text=None, code='custom')"
    var_13 = var_11.__str__()
    assert var_13 == "Message(text=None, code='custom')"
    var_14 = module_0.Position(var_5, var_6, var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no is True
    assert var_14.column_no is True
    assert var_14.char_index == -1511
    var_15 = module_0.Position(var_0, var_8, var_6)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no is None
    assert var_15.column_no == -1511
    assert var_15.char_index is True
    var_16 = var_11.__hash__()
    assert var_16 == -3928249367438130881
    var_17 = var_15.__eq__(var_7)
    assert var_17 is False
    var_18 = var_11.keys()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_18) == 1
    var_19 = var_14.__eq__(var_0)
    assert var_19 is False
    var_20 = module_0.ValidationError(text=var_13, code=var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_20) == 1
    var_21 = var_20.__eq__(var_18)
    assert var_21 is False
    var_22 = var_20.messages()
    var_23 = var_1.__iter__()
    var_24 = var_18.__repr__()
    assert var_24 == 'KeysView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    var_25 = var_14.__eq__(var_14)
    assert var_25 is True
    var_26 = var_11.__iter__()
    var_27 = module_0.Message(text=var_7, position=var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Message'
    assert var_27.text is None
    assert var_27.code == 'custom'
    assert var_27.index == []
    assert f'{type(var_27.start_position).__module__}.{type(var_27.start_position).__qualname__}' == 'builtins.dict_keyiterator'
    assert f'{type(var_27.end_position).__module__}.{type(var_27.end_position).__qualname__}' == 'builtins.dict_keyiterator'
    var_28 = var_27.__eq__(var_18)
    assert var_28 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_17)

def test_case_21():
    var_0 = 'First error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'First error'
    assert var_2.code == 'code1'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'Second error'
    var_4 = 'code2'
    var_5 = 'field'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_3, code=var_4, index=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Second error'
    assert var_7.code == 'code2'
    assert var_7.index == ['field']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_2, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 2
    var_10 = repr(var_9)

def test_case_22():
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
    var_5 = -4
    var_6 = [var_1, var_5]
    var_7 = module_0.Message(text=var_0, index=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == ['users', -4]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_4 == var_7
    assert var_8 is False

def test_case_23():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 5748167320160572150
    var_4 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = var_4.__iter__()
    var_7 = var_4.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = module_0.Message(text=var_4, code=var_4, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_8.text).__module__}.{type(var_8.text).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_8.code).__module__}.{type(var_8.code).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = None
    var_11 = module_0.Message(text=var_7, code=var_1, key=var_7, position=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'ValidationResult(value=None)'
    assert var_11.code == 'custom'
    assert var_11.index == ['ValidationResult(value=None)']
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_11.__repr__()
    assert var_12 == "Message(text='ValidationResult(value=None)', code='custom', index=['ValidationResult(value=None)'])"
    var_13 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_10, position=var_0, messages=var_13)

def test_case_24():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True
    var_5 = None
    var_6 = module_0.Message(text=var_2, key=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == [True]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = -1511
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text='ValidationResult(value=None)', code='custom', index=[True])"
    var_9 = var_3.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = module_0.BaseError(text=var_9, position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_10.__str__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = module_0.Position(var_0, var_0, var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is None
    assert var_12.column_no is None
    assert var_12.char_index is None
    var_13 = var_10.__str__()
    assert var_13 == "Message(text=None, code='custom')"
    var_14 = var_10.__eq__(var_11)
    assert var_14 is False
    var_15 = module_0.Position(var_0, var_7, var_4)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no is None
    assert var_15.column_no == -1511
    assert var_15.char_index is True
    var_16 = var_15.__eq__(var_5)
    assert var_16 is False
    var_17 = var_10.keys()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_17) == 1
    var_18 = var_17.__repr__()
    assert var_18 == 'KeysView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'

@pytest.mark.xfail(strict=True)
def test_case_25():
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
    module_0.ParseError(code=var_5, key=var_0, messages=var_4)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = None
    var_2 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error is None
    var_3 = var_2.__iter__()
    var_4 = var_2.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = module_0.Message(text=var_2, code=var_2, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_5.code).__module__}.{type(var_5.code).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_2.__bool__()
    assert var_6 is True
    var_7 = None
    var_8 = module_0.Message(text=var_4, code=var_1, key=var_4, position=var_1)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'ValidationResult(value=None)'
    assert var_8.code == 'custom'
    assert var_8.index == ['ValidationResult(value=None)']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__repr__()
    assert var_9 == "Message(text='ValidationResult(value=None)', code='custom', index=['ValidationResult(value=None)'])"
    var_10 = [var_8, var_8, var_5, var_5]
    var_11 = module_0.BaseError(position=var_7, messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = var_11.__str__()
    assert var_12 == "{'ValidationResult(value=None)': 'ValidationResult(value=None)', '': ValidationResult(value=None)}"
    var_13 = var_11.__str__()
    assert var_13 == "{'ValidationResult(value=None)': 'ValidationResult(value=None)', '': ValidationResult(value=None)}"
    var_11.__getitem__(var_1)

def test_case_27():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_3, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == [False]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = -1511
    var_8 = var_4.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_6.__eq__(var_4)
    assert var_9 is False
    var_10 = var_1.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_7)

def test_case_28():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_3, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == [False]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.BaseError(text=var_7, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_6.__hash__()
    assert var_10 == 2425399594201038480
    var_11 = var_8.__iter__()
    var_12 = var_11.__eq__(var_11)
    assert var_12 is True
    var_13 = var_11.__iter__()
    var_14 = '-_lJd;)^Q{nX*Rm'
    var_15 = 'B"'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_14, key=var_15, index=var_13)

def test_case_29():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_3, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == [False]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.BaseError(text=var_7, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_9.__str__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_8.__eq__(var_8)
    assert var_11 is False
    var_12 = var_8.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_8.__iter__()
    var_14 = var_8.__len__()
    assert var_14 == 1
    var_15 = var_13.__iter__()
    var_16 = [var_13, var_6]
    with pytest.raises(AssertionError):
        module_0.Message(text=var_15, code=var_15, key=var_15, position=var_16, start_position=var_15, end_position=var_12)

def test_case_30():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_1.__bool__()
    assert var_5 is True
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = None
    var_8 = module_0.Message(text=var_3, key=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'ValidationResult(value=None)'
    assert var_8.code == 'custom'
    assert var_8.index == [True]
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = -1511
    var_10 = var_8.__repr__()
    assert var_10 == "Message(text='ValidationResult(value=None)', code='custom', index=[True])"
    var_11 = var_4.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = module_0.BaseError(text=var_11, position=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    var_13 = var_8.__repr__()
    assert var_13 == "Message(text='ValidationResult(value=None)', code='custom', index=[True])"
    var_14 = var_12.__str__()
    assert var_14 == "Message(text=None, code='custom')"
    var_15 = var_12.messages()
    var_16 = module_0.Position(var_5, var_6, var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no is True
    assert var_16.column_no is True
    assert var_16.char_index == -1511
    var_17 = module_0.Position(var_0, var_9, var_6)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Position'
    assert var_17.line_no is None
    assert var_17.column_no == -1511
    assert var_17.char_index is True
    var_18 = var_17.__eq__(var_7)
    assert var_18 is False
    var_19 = var_12.keys()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_19) == 1
    var_20 = var_16.__eq__(var_0)
    assert var_20 is False
    var_21 = module_0.ValidationError(text=var_14, code=var_0)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_21) == 1
    var_22 = var_21.messages()
    var_23 = var_19.__repr__()
    assert var_23 == 'KeysView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    var_24 = var_16.__eq__(var_16)
    assert var_24 is True
    var_25 = var_12.__iter__()
    var_26 = 2322
    with pytest.raises(AssertionError):
        module_0.Message(text=var_19, key=var_26, index=var_0, position=var_25, end_position=var_16)

def test_case_31():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = None
    var_7 = module_0.Message(text=var_3, key=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'ValidationResult(value=None)'
    assert var_7.code == 'custom'
    assert var_7.index == [False]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_4.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.BaseError(text=var_8, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_10.__str__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = var_9.__eq__(var_9)
    assert var_12 is False
    var_13 = var_9.keys()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13) == 1
    var_14 = var_9.__iter__()
    var_15 = var_11.__eq__(var_14)
    var_16 = var_13.__len__()
    assert var_16 == 1
    var_17 = var_9.__iter__()
    var_18 = module_0.Message(text=var_13, position=var_0, end_position=var_14)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_18.text).__module__}.{type(var_18.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_18.text) == 1
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert var_18.start_position is None
    assert f'{type(var_18.end_position).__module__}.{type(var_18.end_position).__qualname__}' == 'builtins.dict_keyiterator'
    var_19 = var_9.__iter__()
    var_20 = var_19.__eq__(var_6)
    var_21 = var_14.__repr__()
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_17, messages=var_19)

def test_case_32():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True
    var_6 = None
    var_7 = -1511
    var_8 = var_3.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.BaseError(text=var_8, position=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_9.__iter__()
    var_12 = var_9.__eq__(var_10)
    assert var_12 is False
    var_13 = module_0.Position(var_0, var_7, var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no == -1511
    assert var_13.char_index is True
    var_14 = var_13.__eq__(var_6)
    assert var_14 is False
    var_15 = var_9.keys()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_15) == 1
    var_16 = module_0.ValidationError(text=var_11, code=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16) == 1
    var_17 = var_16.messages()
    var_18 = var_15.__repr__()
    assert var_18 == 'KeysView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    var_19 = 194
    var_20 = var_12.__eq__(var_12)
    assert var_20 is True
    var_21 = var_9.__iter__()
    var_22 = var_15.__contains__(var_7)
    assert var_22 is False
    var_23 = var_11.__eq__(var_22)
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_6, key=var_19, messages=var_15)

def test_case_33():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.Message(text=var_3, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == [False]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.BaseError(text=var_7, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__iter__()
    var_10 = var_8.__str__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_8.__eq__(var_8)
    assert var_11 is False
    var_12 = var_8.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = module_0.Position(var_0, var_5, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no is False
    assert f'{type(var_13.char_index).__module__}.{type(var_13.char_index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13.char_index) == 1
    var_14 = var_8.__iter__()
    var_15 = var_13.__eq__(var_0)
    assert var_15 is False
    var_16 = var_12.__contains__(var_0)
    assert var_16 is False
    var_17 = module_0.Message(text=var_10, start_position=var_14)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == "Message(text=None, code='custom')"
    assert var_17.code == 'custom'
    assert var_17.index == []
    assert f'{type(var_17.start_position).__module__}.{type(var_17.start_position).__qualname__}' == 'builtins.dict_keyiterator'
    assert var_17.end_position is None
    var_18 = var_13.__repr__()
    assert var_18 == 'Position(line_no=None, column_no=False, char_index=KeysView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\')))'
    var_19 = var_14.__eq__(var_14)
    assert var_19 is True
    var_20 = None
    var_21 = var_17.__repr__()
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_20)

def test_case_34():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = module_0.Message(text=var_3, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'ValidationResult(value=None)'
    assert var_6.code == 'custom'
    assert var_6.index == [True]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.BaseError(text=var_7, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_9.__str__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = var_8.__eq__(var_8)
    assert var_11 is False
    var_12 = var_8.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_8.__iter__()
    var_14 = var_12.__repr__()
    assert var_14 == 'KeysView(BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_13, error=var_12)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = None
    var_8 = module_0.Message(text=var_6, key=var_0, end_position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is True
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__repr__()
    assert var_9 == "Message(text=True, code='custom')"
    var_10 = var_8.__repr__()
    assert var_10 == "Message(text=True, code='custom')"
    var_11 = module_0.BaseError(text=var_10, key=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_11.__str__()
    assert var_12 == '{"Message(text=True, code=\'custom\')": "Message(text=True, code=\'custom\')"}'
    var_13 = var_11.values()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_13) == 1
    var_14 = var_13.__str__()
    assert var_14 == 'ValuesView(BaseError([Message(text="Message(text=True, code=\'custom\')", code=\'custom\', index=["Message(text=True, code=\'custom\')"])]))'
    var_15 = var_11.__eq__(var_0)
    assert var_15 is False
    var_16 = False
    var_17 = False
    var_18 = module_0.Position(var_0, var_16, var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Position'
    assert var_18.line_no is None
    assert var_18.column_no is False
    assert var_18.char_index is False
    var_19 = var_15.__eq__(var_0)
    var_13.keys()

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = 'M~FPx3PT=bd\x0c![z'
    var_5 = module_0.Message(text=var_4, position=var_0, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'M~FPx3PT=bd\x0c![z'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_1)
    assert var_6 is False
    var_7 = var_5.__eq__(var_0)
    assert var_7 is False
    var_8 = None
    var_9 = module_0.Message(text=var_0, index=var_0, start_position=var_8, end_position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is None
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = -1511
    var_11 = var_9.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = var_9.__repr__()
    assert var_12 == "Message(text=None, code='custom')"
    var_13 = '/'
    var_14 = module_0.BaseError(text=var_3, code=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_14.__str__()
    assert var_15 == 'ValidationResult(value=None)'
    var_16 = var_14.__str__()
    assert var_16 == 'ValidationResult(value=None)'
    var_17 = var_14.__eq__(var_8)
    assert var_17 is False
    var_18 = True
    var_19 = module_0.Position(var_18, var_7, var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no is True
    assert var_19.column_no is False
    assert var_19.char_index is None
    var_20 = var_17.__eq__(var_0)
    var_21 = var_14.keys()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_21) == 1
    var_22 = var_14.values()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_22) == 1
    var_23 = var_14.messages(add_prefix=var_10)
    var_24 = var_21.__repr__()
    assert var_24 == "KeysView(BaseError(text='ValidationResult(value=None)', code='/'))"
    var_25 = var_19.__eq__(var_0)
    assert var_25 is False
    var_26 = var_21.__iter__()
    var_26.__contains__(var_26)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = None
    var_1 = b'\xff\x1f\x10_\\\x18K\x14\xd0\xd6\x87)\xdc5Z'
    var_2 = module_0.ValidationResult(value=var_0, error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error == b'\xff\x1f\x10_\\\x18K\x14\xd0\xd6\x87)\xdc5Z'
    var_3 = var_2.__iter__()
    var_4 = var_2.__repr__()
    assert var_4 == "ValidationResult(error=b'\\xff\\x1f\\x10_\\\\\\x18K\\x14\\xd0\\xd6\\x87)\\xdc5Z')"
    var_5 = module_0.Message(text=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    module_0.ValidationError()

def test_case_38():
    var_0 = 0
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 0
    assert var_1.column_no == 0
    assert var_1.char_index == 0
    var_2 = 2
    var_3 = 10
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 2
    assert var_4.column_no == 10
    assert var_4.char_index == 10
    var_5 = 'Error'
    var_6 = module_0.Message(text=var_5, start_position=var_1, end_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = module_0.Message(text=var_5, start_position=var_4, end_position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = var_6 == var_7
    assert var_8 is False

def test_case_39():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = var_1.__repr__()
    assert var_3 == 'ValidationResult(value=None)'
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_1.__bool__()
    assert var_5 is True
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True
    var_7 = None
    var_8 = module_0.Message(text=var_3, key=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'ValidationResult(value=None)'
    assert var_8.code == 'custom'
    assert var_8.index == [True]
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = -1511
    var_10 = var_8.__repr__()
    assert var_10 == "Message(text='ValidationResult(value=None)', code='custom', index=[True])"
    var_11 = var_4.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = module_0.BaseError(text=var_11, position=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    var_13 = var_12.__str__()
    assert var_13 == "Message(text=None, code='custom')"
    var_14 = var_12.__str__()
    assert var_14 == "Message(text=None, code='custom')"
    var_15 = module_0.Position(var_5, var_6, var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no is True
    assert var_15.column_no is True
    assert var_15.char_index == -1511
    var_16 = module_0.Position(var_0, var_9, var_6)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no is None
    assert var_16.column_no == -1511
    assert var_16.char_index is True
    var_17 = var_16.__eq__(var_7)
    assert var_17 is False
    var_18 = var_12.keys()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_18) == 1
    var_19 = module_0.ValidationError(text=var_4, key=var_7)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19) == 1
    var_20 = var_12.messages(add_prefix=var_0)
    var_21 = var_1.__repr__()
    assert var_21 == 'ValidationResult(value=None)'
    var_22 = var_19.__eq__(var_19)
    assert var_22 is True
    var_23 = var_15.__eq__(var_9)
    assert var_23 is False
    var_24 = var_18.__iter__()
    var_25 = module_0.Message(text=var_21)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text == 'ValidationResult(value=None)'
    assert var_25.code == 'custom'
    assert var_25.index == []
    assert var_25.start_position is None
    assert var_25.end_position is None
    var_26 = var_24.__eq__(var_0)
    var_27 = var_19.__contains__(var_24)
    assert var_27 is False

def test_case_40():
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

def test_case_41():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 1'
    assert var_6.code == 'code1'
    assert var_6.index == ['users', 0, 'name']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = 1
    var_10 = 'email'
    var_11 = [var_2, var_9, var_10]
    var_12 = module_0.Message(text=var_7, code=var_8, index=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'Error 2'
    assert var_12.code == 'code2'
    assert var_12.index == ['users', 1, 'email']
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = [var_6, var_12]
    var_14 = module_0.BaseError(messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_14._messages
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = dict(var_14)