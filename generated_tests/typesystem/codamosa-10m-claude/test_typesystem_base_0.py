# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is False
    assert var_2.char_index is False
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
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
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_5():
    with pytest.raises(AssertionError):
        module_0.BaseError()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    module_0.BaseError(position=var_2, messages=var_3)

def test_case_7():
    var_0 = None
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = var_1.__repr__()
    assert var_2 == 'True'
    var_3 = module_0.BaseError(text=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1

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
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__repr__()
    assert var_4 == "BaseError(text=True, code='custom')"
    var_5 = var_3.__iter__()
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_4, code=var_4, messages=var_5)

def test_case_9():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = ''
    var_2 = module_0.ParseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1
    var_3.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_4.__getitem__(var_0)

def test_case_12():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'j9Bc'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    module_0.ParseError(code=var_0)

def test_case_14():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_1.__hash__()
    assert var_5 == -3414395154972785824

def test_case_15():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.__len__()
    assert var_5 == 1
    var_6 = var_4.__contains__(var_4)
    assert var_6 is False
    var_7 = var_4.__eq__(var_6)
    assert var_7 is False

def test_case_16():
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
    var_3 = module_0.Message(text=var_0, key=var_2, start_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == ["Message(text=None, code='custom')"]
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1.__eq__(var_0)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages()
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_4.__iter__()
    var_8 = var_4.__len__()
    assert var_8 == 1
    var_9 = var_4.__contains__(var_7)
    assert var_9 is False
    module_0.ValidationError(code=var_8, position=var_9, messages=var_9)

def test_case_18():
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
    var_4 = var_3.__repr__()
    assert var_4 == 'True'
    var_5 = module_0.BaseError(text=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "BaseError(text=True, code='custom')"
    var_7 = var_5.__iter__()
    var_8 = var_7.__eq__(var_2)

def test_case_19():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__bool__()
    assert var_2 is True
    var_3 = module_0.BaseError(text=var_1, key=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_0, position=var_1, end_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert f'{type(var_4.start_position).__module__}.{type(var_4.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.end_position).__module__}.{type(var_4.end_position).__qualname__}' == 'typesystem.base.ValidationResult'
    var_5 = var_1.__hash__()
    var_6 = var_3.__str__()
    assert var_6 == '{ValidationResult(value=None): ValidationResult(value=None)}'
    var_7 = var_3.__len__()
    assert var_7 == 1

def test_case_20():
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
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False

def test_case_21():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'
    var_2 = var_0.__iter__()

def test_case_22():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages()
    var_6 = var_4.__repr__()
    assert var_6 == "BaseError(text=True, code='custom')"
    var_7 = var_4.__iter__()
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_3, messages=var_7)

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
    var_3 = module_0.BaseError(text=var_2, position=var_1, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError()

@pytest.mark.xfail(strict=True)
def test_case_24():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages()
    var_6 = var_4.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6) == 1
    var_7 = var_4.__iter__()
    var_8 = var_7.__repr__()
    var_9 = var_7.__str__()
    module_0.BaseError(text=var_9, code=var_6, key=var_6, position=var_6)

def test_case_25():
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
    var_3 = var_2.__repr__()
    assert var_3 == 'True'
    var_4 = module_0.BaseError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages()
    var_6 = var_4.__repr__()
    assert var_6 == "BaseError(text=True, code='custom')"
    var_7 = var_4.__iter__()
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_7, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = False
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is False
    assert var_1.error is None
    var_2 = None
    module_0.ValidationError(position=var_2)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = None
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=None, column_no=None, char_index=None)'
    module_0.ValidationError(position=var_1)

def test_case_28():
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
    var_5 = var_4.__repr__()
    assert var_5 == "BaseError(text=True, code='custom')"
    var_6 = var_1.__eq__(var_0)
    assert var_6 is False
    var_7 = var_4.__len__()
    assert var_7 == 1
    var_8 = var_4.__str__()
    assert var_8 is True
    var_9 = var_1.__repr__()
    assert var_9 == "Message(text=None, code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_29():
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
    var_3 = [var_1, var_1, var_1, var_1]
    var_4 = module_0.BaseError(messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "BaseError([Message(text=None, code='custom'), Message(text=None, code='custom'), Message(text=None, code='custom'), Message(text=None, code='custom')])"
    var_6 = var_4.values()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_6) == 1
    var_6.messages()

def test_case_30():
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
    var_5 = module_0.BaseError(text=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_4.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_5.__iter__()
    var_9 = var_8.__iter__()
    var_10 = var_9.__repr__()
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_8, error=var_9)

def test_case_31():
    var_0 = None
    var_1 = 'yvA6ia;"7l2Rsy%'
    var_2 = module_0.Message(text=var_1, code=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'yvA6ia;"7l2Rsy%'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == 'Message(text=\'yvA6ia;"7l2Rsy%\', code=\'custom\')'
    var_5 = module_0.BaseError(text=var_1, code=var_1, key=var_1, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_5.__repr__()
    assert var_7 == 'BaseError([Message(text=\'yvA6ia;"7l2Rsy%\', code=\'yvA6ia;"7l2Rsy%\', index=[\'yvA6ia;"7l2Rsy%\'])])'
    var_8 = var_2.__eq__(var_0)
    assert var_8 is False
    var_9 = var_5.__len__()
    assert var_9 == 1
    var_10 = var_5.__str__()
    assert var_10 == '{\'yvA6ia;"7l2Rsy%\': \'yvA6ia;"7l2Rsy%\'}'
    var_11 = var_5.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_0, position=var_11)

@pytest.mark.xfail(strict=True)
def test_case_32():
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
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.messages(add_prefix=var_2)
    var_5 = var_3.__repr__()
    assert var_5 == "BaseError(text=True, code='custom')"
    var_6 = var_3.__str__()
    assert var_6 is True
    var_7 = var_3.__len__()
    assert var_7 == 1
    var_7.values()

def test_case_33():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.BaseError(text=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_0, position=var_1, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.ValidationResult'
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text=None, code='custom', position=ValidationResult(value=None))"
    var_5 = var_1.__hash__()

def test_case_34():
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
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.BaseError(text=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = module_0.Message(text=var_0, key=var_4, start_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == ["Message(text=None, code='custom')"]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_5.messages()
    var_8 = var_5.__repr__()
    assert var_8 == "BaseError(text=True, code='custom')"
    var_9 = var_5.__contains__(var_6)
    assert var_9 is False
    var_10 = var_1.__eq__(var_0)
    assert var_10 is False
    var_11 = var_5.__len__()
    assert var_11 == 1
    var_12 = var_5.__str__()
    assert var_12 is True
    var_13 = module_0.BaseError(messages=var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = module_0.ValidationError(text=var_9, key=var_11, position=var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_5.__hash__()
    assert var_15 == 6913292012659488474
    var_16 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_0, messages=var_16)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_5 = module_0.Message(text=var_0, key=var_3, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == ["Message(text=None, code='custom')"]
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_4.messages()
    var_7 = var_4.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_4.__contains__(var_0)
    assert var_8 is False
    var_9 = var_1.__eq__(var_5)
    assert var_9 is False
    var_10 = var_4.__len__()
    assert var_10 == 1
    var_11 = var_8.__str__()
    assert var_11 == 'False'
    var_12 = module_0.BaseError(text=var_8, key=var_8, position=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    module_0.ValidationError(position=var_0)

def test_case_36():
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
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index is None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.BaseError(text=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_5.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_1.__eq__(var_0)
    assert var_8 is False
    var_9 = var_5.keys()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_9) == 1
    var_10 = var_5.__len__()
    assert var_10 == 1
    var_11 = var_5.__str__()
    assert var_11 is True
    var_12 = module_0.BaseError(messages=var_6)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 1
    var_13 = var_5.__eq__(var_0)
    assert var_13 is False
    var_14 = module_0.BaseError(text=var_13, key=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_12.items()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_15) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_15, code=var_13, key=var_2, index=var_13, start_position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_37():
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
    var_3 = module_0.Position(var_2, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index is None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.BaseError(text=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.messages()
    var_7 = var_5.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_5.__iter__()
    var_9 = var_5.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = var_5.__str__()
    assert var_10 is True
    var_11 = module_0.BaseError(messages=var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = var_5.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.BaseError(text=var_12, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_11.__hash__()
    assert var_14 == 6913292012659488474
    var_15 = module_0.Message(text=var_0, position=var_12, end_position=var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text is None
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert var_15.start_position is False
    assert var_15.end_position is False
    var_16 = var_15.__eq__(var_1)
    assert var_16 is False
    module_0.ValidationError(key=var_8)

def test_case_38():
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
    var_5 = var_4.messages()
    var_6 = var_4.__repr__()
    assert var_6 == "BaseError(text=True, code='custom')"
    var_7 = var_4.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_4.__str__()
    assert var_8 is True
    var_9 = var_1.__eq__(var_0)
    assert var_9 is False
    var_10 = var_4.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_4.__len__()
    assert var_11 == 1
    var_12 = var_4.__str__()
    assert var_12 is True
    var_13 = module_0.BaseError(messages=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_4.__eq__(var_0)
    assert var_14 is False
    var_15 = module_0.BaseError(text=var_14, key=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = var_10.__repr__()
    assert var_16 == "KeysView(BaseError(text=True, code='custom'))"
    var_17 = var_13.items()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17) == 1
    var_18 = var_13.__hash__()
    assert var_18 == 6913292012659488474
    var_19 = module_0.Message(text=var_0, position=var_14, end_position=var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is None
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is False
    assert var_19.end_position is False
    var_20 = 'j/RrKa^8+'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_20, position=var_17, end_position=var_14)

@pytest.mark.xfail(strict=True)
def test_case_39():
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
    var_5 = var_4.messages()
    var_6 = var_4.__repr__()
    assert var_6 == "BaseError(text=True, code='custom')"
    var_7 = var_4.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_4.__str__()
    assert var_8 is True
    var_9 = var_1.__eq__(var_0)
    assert var_9 is False
    var_10 = var_4.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_4.__len__()
    assert var_11 == 1
    var_12 = var_4.__str__()
    assert var_12 is True
    var_13 = module_0.BaseError(messages=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = var_4.__eq__(var_0)
    assert var_14 is False
    var_15 = module_0.BaseError(text=var_14, key=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = var_10.__repr__()
    assert var_16 == "KeysView(BaseError(text=True, code='custom'))"
    var_17 = var_13.__hash__()
    assert var_17 == 6913292012659488474
    var_18 = module_0.Message(text=var_0, position=var_14, end_position=var_0)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text is None
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert var_18.start_position is False
    assert var_18.end_position is False
    var_19 = var_18.__eq__(var_14)
    assert var_19 is False
    var_20 = module_0.Message(text=var_10, start_position=var_14)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_20.text).__module__}.{type(var_20.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_20.text) == 1
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert var_20.start_position is False
    assert var_20.end_position is None
    var_21 = var_14.__eq__(var_12)
    assert var_21 is False
    var_22 = var_20.__repr__()
    assert var_22 == "Message(text=KeysView(BaseError(text=True, code='custom')), code='custom', start_position=False, end_position=None)"
    var_23 = var_4.__hash__()
    assert var_23 == 6913292012659488474
    var_13.__getitem__(var_10)

def test_case_40():
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
    var_5 = var_4.messages()
    var_6 = var_4.__repr__()
    assert var_6 == "BaseError(text=True, code='custom')"
    var_7 = var_4.__repr__()
    assert var_7 == "BaseError(text=True, code='custom')"
    var_8 = var_4.__str__()
    assert var_8 is True
    var_9 = var_4.__contains__(var_1)
    assert var_9 is False
    var_10 = var_4.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_4.__len__()
    assert var_11 == 1
    var_12 = var_4.__str__()
    assert var_12 is True
    var_13 = module_0.BaseError(messages=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 1
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = module_0.BaseError(text=var_14, key=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = var_13.items()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16) == 1
    var_17 = var_4.__hash__()
    assert var_17 == 6913292012659488474
    var_18 = 'd,tuWdz0%85RVKY'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_18, position=var_10, start_position=var_9)

def test_case_41():
    var_0 = 'Error message'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = 'Different message'
    var_5 = module_0.ValidationError(text=var_4, code=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = 'different_code'
    var_7 = module_0.ValidationError(text=var_0, code=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = 'Error 1'
    var_9 = 'code1'
    var_10 = 'field1'
    var_11 = module_0.Message(text=var_8, code=var_9, key=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'Error 1'
    assert var_11.code == 'code1'
    assert var_11.index == ['field1']
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = 'Error 2'
    var_13 = 'code2'
    var_14 = 'field2'
    var_15 = module_0.Message(text=var_12, code=var_13, key=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'Error 2'
    assert var_15.code == 'code2'
    assert var_15.index == ['field2']
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = [var_11, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17) == 2
    var_18 = module_0.ValidationError(messages=var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18) == 2
    var_19 = module_0.Message(text=var_8, code=var_9, key=var_10)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text == 'Error 1'
    assert var_19.code == 'code1'
    assert var_19.index == ['field1']
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = 'Different Error'
    var_21 = module_0.Message(text=var_20, code=var_13, key=var_14)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text == 'Different Error'
    assert var_21.code == 'code2'
    assert var_21.index == ['field2']
    assert var_21.start_position is None
    assert var_21.end_position is None
    var_22 = [var_19, var_21]
    var_23 = module_0.ValidationError(messages=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_23) == 2
    var_24 = module_0.ParseError(text=var_0, code=var_1)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_24) == 1
    var_25 = 1
    var_26 = 5
    var_27 = 4
    var_28 = module_0.Position(var_25, var_26, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Position'
    assert var_28.line_no == 1
    assert var_28.column_no == 5
    assert var_28.char_index == 4
    var_29 = 'Error with position'
    var_30 = 'pos_code'
    var_31 = module_0.ValidationError(text=var_29, code=var_30, position=var_28)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_31) == 1
    var_32 = 2
    var_33 = 10
    var_34 = 9
    var_35 = module_0.Position(var_32, var_33, var_34)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.base.Position'
    assert var_35.line_no == 2
    assert var_35.column_no == 10
    assert var_35.char_index == 9
    var_36 = module_0.ValidationError(text=var_29, code=var_30, position=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_36) == 1
    var_37 = 'Nested error'
    var_38 = 'nested'
    var_39 = 'users'
    var_40 = 0
    var_41 = 'name'
    var_42 = [var_39, var_40, var_41]
    var_43 = module_0.Message(text=var_37, code=var_38, index=var_42)
    assert f'{type(var_43).__module__}.{type(var_43).__qualname__}' == 'typesystem.base.Message'
    assert var_43.text == 'Nested error'
    assert var_43.code == 'nested'
    assert var_43.index == ['users', 0, 'name']
    assert var_43.start_position is None
    assert var_43.end_position is None
    var_44 = [var_43]
    var_45 = module_0.ValidationError(messages=var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_45) == 1
    var_46 = module_0.ValidationError(messages=var_44)
    assert f'{type(var_46).__module__}.{type(var_46).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_46) == 1
    var_47 = [var_39, var_25, var_41]
    var_48 = module_0.Message(text=var_37, code=var_38, index=var_47)
    assert f'{type(var_48).__module__}.{type(var_48).__qualname__}' == 'typesystem.base.Message'
    assert var_48.text == 'Nested error'
    assert var_48.code == 'nested'
    assert var_48.index == ['users', 1, 'name']
    assert var_48.start_position is None
    assert var_48.end_position is None
    var_49 = [var_48]
    var_50 = module_0.ValidationError(messages=var_49)
    assert f'{type(var_50).__module__}.{type(var_50).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_50) == 1

def test_case_42():
    var_0 = 'Test error'
    var_1 = 'test_code'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = repr(var_3)
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.ValidationResult(value=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == {'key': 'value'}
    assert var_8.error is None
    var_9 = repr(var_8)
    assert var_9 == "ValidationResult(value={'key': 'value'})"
    var_10 = module_0.ValidationResult(value=var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == "ValidationResult(error=ValidationError(text='Test error', code='test_code'))"
    assert var_10.error is None
    var_11 = 'test_string'
    var_12 = module_0.ValidationResult(value=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value == 'test_string'
    assert var_12.error is None
    var_13 = repr(var_12)
    assert var_13 == "ValidationResult(value='test_string')"
    var_14 = 42
    var_15 = module_0.ValidationResult(value=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value == 42
    assert var_15.error is None
    var_16 = repr(var_15)
    assert var_16 == 'ValidationResult(value=42)'
    var_17 = 1
    var_18 = 2
    var_19 = 3
    var_20 = [var_17, var_18, var_19]
    var_21 = module_0.ValidationResult(value=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_21.value == [1, 2, 3]
    assert var_21.error is None
    var_22 = repr(var_21)
    assert var_22 == 'ValidationResult(value=[1, 2, 3])'

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 42
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 42
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = module_0.ValidationError(text=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = module_0.ValidationResult(error=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = list(var_5)
    var_7 = len(var_6)
    assert var_7 == 2
    var_3.get(var_2)

@pytest.mark.xfail(strict=True)
def test_case_44():
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
    var_3 = [var_1, var_1, var_1, var_1]
    var_4 = module_0.BaseError(messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.__str__()
    assert var_5 == "{'': None}"
    var_6 = var_4.values()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_6) == 1
    var_6.messages()

def test_case_45():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is False
    assert var_2.char_index is False
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
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
    var_3 = var_2.__bool__()
    assert var_3 is True
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.BaseError(text=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = False
    var_8 = module_0.Position(var_4, var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is False
    assert var_8.char_index is False
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = var_6.messages()
    var_11 = var_6.__repr__()
    assert var_11 == "BaseError(text=True, code='custom')"
    var_12 = var_6.__repr__()
    assert var_12 == "BaseError(text=True, code='custom')"
    var_13 = var_6.__str__()
    assert var_13 is True
    var_14 = var_6.__contains__(var_1)
    assert var_14 is False
    var_15 = var_6.__len__()
    assert var_15 == 1
    var_16 = var_6.__str__()
    assert var_16 is True
    var_17 = module_0.BaseError(messages=var_10)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = module_0.ValidationResult()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_18.value is None
    assert var_18.error is None
    var_19 = var_14.__repr__()
    assert var_19 == 'False'
    var_20 = var_6.messages()
    var_21 = var_8.__eq__(var_8)
    assert var_21 is True
    var_22 = var_14.__repr__()
    assert var_22 == 'False'
    var_23 = var_6.__eq__(var_0)
    assert var_23 is False
    var_24 = var_17.values()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_24) == 1
    var_25 = var_24.__repr__()
    assert var_25 == "ValuesView(BaseError(text=True, code='custom'))"
    var_26 = var_17.__hash__()
    assert var_26 == 6913292012659488474
    var_27 = var_20.__getitem__(var_23)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Message'
    assert var_27.text is True
    assert var_27.code == 'custom'
    assert var_27.index == []
    assert var_27.start_position is None
    assert var_27.end_position is None
    var_28 = var_20.__contains__(var_1)
    assert var_28 is False
    var_14.__len__()

@pytest.mark.xfail(strict=True)
def test_case_47():
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
    var_5 = False
    var_6 = module_0.Position(var_2, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is True
    assert var_6.column_no is False
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = module_0.Position(var_0, var_7, var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is False
    assert var_8.char_index is False
    var_9 = var_6.__eq__(var_8)
    assert var_9 is False
    var_10 = var_4.messages()
    var_11 = var_4.__repr__()
    assert var_11 == "BaseError(text=True, code='custom')"
    var_12 = var_4.__repr__()
    assert var_12 == "BaseError(text=True, code='custom')"
    var_4.__getitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_48():
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
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is True
    assert f'{type(var_4.column_no).__module__}.{type(var_4.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4.column_no) == 1
    assert f'{type(var_4.char_index).__module__}.{type(var_4.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4.char_index) == 1
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False
    var_6 = var_4.__eq__(var_0)
    assert var_6 is False
    var_7 = var_3.messages()
    var_8 = var_3.__repr__()
    assert var_8 == "BaseError(text=True, code='custom')"
    var_9 = var_3.__repr__()
    assert var_9 == "BaseError(text=True, code='custom')"
    var_10 = var_3.__str__()
    assert var_10 is True
    var_11 = var_3.__contains__(var_1)
    assert var_11 is False
    var_12 = var_3.__len__()
    assert var_12 == 1
    var_13 = var_3.__iter__()
    var_14 = var_3.__str__()
    assert var_14 is True
    var_15 = module_0.BaseError(messages=var_7)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15) == 1
    var_16 = module_0.ValidationResult()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_11.__repr__()
    assert var_17 == 'False'
    var_18 = var_15.items()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_18) == 1
    var_19 = var_4.__eq__(var_4)
    assert var_19 is False
    var_20 = var_11.__repr__()
    assert var_20 == 'False'
    var_21 = var_3.__eq__(var_0)
    assert var_21 is False
    var_22 = var_15.values()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_22) == 1
    var_23 = var_22.__repr__()
    assert var_23 == "ValuesView(BaseError(text=True, code='custom'))"
    var_24 = var_15.__hash__()
    assert var_24 == 6913292012659488474
    var_25 = var_15.values()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_25) == 1
    var_15.__getitem__(var_25)