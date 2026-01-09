# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is False
    assert var_2.char_index is False
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_2():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = set()
    var_5 = module_0.BaseError(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = None
    var_7 = module_0.ValidationResult(value=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_2.__hash__()
    assert var_8 == 4004646046084524075

def test_case_3():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

def test_case_4():
    var_0 = "'x\x0b]*;t84"
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.ParseError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = ':$o\tOB?6u"Neu">/'
    var_1 = None
    module_0.ValidationError(position=var_1, messages=var_0)

def test_case_7():
    var_0 = []
    var_1 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_1, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = None
    var_6 = module_0.BaseError(text=var_2, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_0)
    assert var_7 is False
    var_8 = var_6.messages()
    var_9 = var_6.values()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_9) == 1
    var_10 = module_0.Position(var_5, var_7, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is False
    assert f'{type(var_10.char_index).__module__}.{type(var_10.char_index).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_10.char_index) == 1
    var_11 = var_6.__eq__(var_4)
    assert var_11 is False
    var_4.messages()

def test_case_9():
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
    var_6 = module_0.ValidationError(text=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult(error=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = iter(var_7)
    var_9 = next(var_8)
    assert var_9 is None
    var_10 = next(var_8)
    var_11 = "KxnEo'QJ_3+FBjpj."
    var_12 = print(var_11)

def test_case_10():
    var_0 = '\nO\trB<e|]KNg0|@9('
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error == '\nO\trB<e|]KNg0|@9('

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 4004646046084524075
    module_0.ParseError(text=var_0, code=var_0, key=var_0)

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
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = None
    var_6 = module_0.BaseError(text=var_2, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_0)
    assert var_7 is False
    var_8 = var_6.messages()
    var_9 = var_6.__len__()
    assert var_9 == 1
    var_10 = module_0.Position(var_5, var_7, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is False
    assert var_10.char_index == 1
    var_11 = var_6.__eq__(var_4)
    assert var_11 is False
    var_4.messages()

def test_case_13():
    var_0 = "'x\x0b]*;t84"
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_3.__bool__()
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = None
    var_2 = var_0.__bool__()
    assert var_2 is True
    module_0.ParseError(code=var_1)

def test_case_15():
    var_0 = 'es'
    var_1 = 2
    var_2 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 2
    assert var_2.column_no == 2
    assert var_2.char_index == 2
    var_3 = module_0.Message(text=var_0, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'es'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.Position'

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == ['test']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 2
    var_3 = module_0.Position(var_2, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 2
    assert var_3.column_no == 2
    assert var_3.char_index == 2
    var_4 = None
    var_5 = var_3.__eq__(var_4)
    assert var_5 is False
    var_6 = module_0.Message(text=var_0, position=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'test'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.Position'
    var_7 = module_0.Position(var_5, var_2, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is False
    assert var_7.column_no == 2
    assert var_7.char_index is False
    module_0.ValidationError(text=var_4, key=var_4, position=var_4, messages=var_4)

def test_case_17():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = None
    var_6 = module_0.BaseError(text=var_2, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_5)
    assert var_7 is False
    var_8 = var_1.__eq__(var_0)
    assert var_8 is False
    var_9 = var_6.messages()
    var_10 = var_6.__len__()
    assert var_10 == 1
    var_11 = module_0.Position(var_0, var_8, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert var_11.column_no is False
    assert var_11.char_index == 1
    var_12 = var_6.__str__()
    assert var_12 == "Message(text=None, code='custom')"
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_10, error=var_10)

def test_case_18():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = None
    var_5 = None
    var_6 = set()
    var_7 = module_0.BaseError(text=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_3.__bool__()
    assert var_8 is True
    var_9 = var_7.__contains__(var_4)
    assert var_9 is False
    var_10 = var_7.__eq__(var_5)
    assert var_10 is False
    var_11 = var_7.__hash__()
    assert var_11 == 5346322630058117098
    var_12 = var_10.__bool__()
    assert var_12 is False
    var_10.__getitem__(var_4)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = None
    var_6 = module_0.BaseError(text=var_2, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_5)
    assert var_7 is False
    var_8 = var_1.__eq__(var_0)
    assert var_8 is False
    var_9 = None
    var_10 = var_6.messages()
    var_11 = var_6.__len__()
    assert var_11 == 1
    var_12 = module_0.Position(var_9, var_8, var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is None
    assert var_12.column_no is False
    assert var_12.char_index == 1
    var_13 = -821.231122
    var_14 = var_11.__eq__(var_13)
    var_15 = var_6.__iter__()
    var_11.messages()

def test_case_21():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = True
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = None
    var_6 = module_0.BaseError(text=var_4, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_5)
    assert var_7 is False
    var_8 = var_1.__eq__(var_5)
    assert var_8 is False
    var_9 = None
    var_10 = var_6.messages(add_prefix=var_2)
    var_11 = var_6.__eq__(var_9)
    assert var_11 is False
    var_12 = var_6.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_6.get(var_5)
    var_14 = var_6.messages()
    var_15 = var_12.__repr__()
    assert var_15 == "KeysView(BaseError(text='ValidationResult(value=None)', code='custom'))"

def test_case_22():
    var_0 = 'max_length'
    var_1 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'max_length'
    assert var_1.code == 'custom'
    assert var_1.index == ['max_length']
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'users'
    var_3 = 1
    var_4 = 2
    var_5 = None
    var_6 = var_1.__eq__(var_5)
    assert var_6 is False
    var_7 = module_0.Position(var_3, var_4, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 2
    assert var_7.char_index is False
    var_8 = 5
    var_9 = 6
    var_10 = module_0.Position(var_3, var_8, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 5
    assert var_10.char_index == 6
    var_11 = module_0.Message(text=var_2, start_position=var_7, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'users'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.Position'
    var_12 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'max_length'
    assert var_12.code == 'max_length'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = module_0.Position(var_3, var_4, var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 1
    assert var_13.column_no == 2
    assert var_13.char_index == 1
    var_14 = module_0.Message(text=var_0, position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'max_length'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert f'{type(var_14.start_position).__module__}.{type(var_14.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'typesystem.base.Position'
    var_15 = repr(var_14)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = None
    var_6 = module_0.BaseError(text=var_2, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_1.__eq__(var_0)
    assert var_7 is False
    var_8 = None
    var_9 = var_6.messages()
    var_10 = var_6.__len__()
    assert var_10 == 1
    var_11 = module_0.Position(var_8, var_7, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert var_11.column_no is False
    assert var_11.char_index == 1
    module_0.ValidationError(position=var_11, messages=var_9)

def test_case_24():
    var_0 = "'x\x0b]*;t84"
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_0, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "'x\x0b]*;t84"
    assert var_2.code == "'x\x0b]*;t84"
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 2
    var_4 = False
    var_5 = module_0.Position(var_3, var_4, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 2
    assert var_5.column_no is False
    assert var_5.char_index is None
    var_6 = None
    var_7 = var_5.__eq__(var_6)
    assert var_7 is False
    var_8 = var_2.__eq__(var_1)
    assert var_8 is False
    var_9 = module_0.Message(text=var_0, code=var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == "'x\x0b]*;t84"
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = True
    var_11 = module_0.Position(var_6, var_5, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert f'{type(var_11.column_no).__module__}.{type(var_11.column_no).__qualname__}' == 'typesystem.base.Position'
    assert var_11.char_index is True
    var_12 = 5
    var_13 = module_0.ValidationError(text=var_8, key=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1

def test_case_25():
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
    var_6 = module_0.ValidationError(text=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult(error=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = iter(var_7)
    var_9 = var_7.__repr__()
    assert var_9 == "ValidationResult(error=ValidationError(text='test_error', code='custom'))"
    var_10 = next(var_8)
    assert var_10 is None
    var_11 = next(var_8)
    var_12 = 'All test cases passed'
    var_13 = print(var_12)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = -1126
    var_5 = 1448
    var_6 = module_0.Position(var_4, var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == -1126
    assert var_6.column_no == 1448
    assert var_6.char_index is None
    var_7 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_6.__eq__(var_0)
    assert var_8 is False
    var_9 = var_1.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = var_2.__eq__(var_0)
    assert var_10 is False
    var_11 = var_2.__hash__()
    assert var_11 == 4004646046084524075
    module_0.ParseError(key=var_1, messages=var_1)

def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == 'ValidationResult(value=None)'
    var_5 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'test'
    assert var_3.code == 'custom'
    assert var_3.index == ['test']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = -26
    var_5 = -21
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text='test', code='custom')"
    var_7 = None
    var_8 = var_3.__eq__(var_7)
    assert var_8 is False
    var_9 = 5
    var_10 = 6
    var_11 = module_0.Position(var_4, var_9, var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no == -26
    assert var_11.column_no == 5
    assert var_11.char_index == 6
    var_12 = module_0.Message(text=var_0, start_position=var_3, end_position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'test'
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert f'{type(var_12.start_position).__module__}.{type(var_12.start_position).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_12.end_position).__module__}.{type(var_12.end_position).__qualname__}' == 'typesystem.base.Position'
    var_13 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'test'
    assert var_13.code == 'max_length'
    assert var_13.index == []
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = hash(var_13)
    var_15 = hash(var_13)
    var_16 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'test'
    assert var_16.code == 'max_length'
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = module_0.Message(text=var_0, index=var_10)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'test'
    assert var_17.code == 'custom'
    assert var_17.index == 6
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = repr(var_17)
    var_19 = module_0.Position(var_4, var_5, var_4)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no == -26
    assert var_19.column_no == -21
    assert var_19.char_index == -26
    var_20 = module_0.Message(text=var_0, position=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'test'
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert f'{type(var_20.start_position).__module__}.{type(var_20.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_20.end_position).__module__}.{type(var_20.end_position).__qualname__}' == 'typesystem.base.Position'
    var_21 = repr(var_20)
    module_0.ParseError(messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = ':$o\tOgsB?6u"N5u">/'
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = True
    var_4 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = module_0.ValidationError(text=var_4, key=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.values()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_6) == 1
    var_7 = var_4.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = None
    var_9 = module_0.BaseError(text=var_6, position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_2.__eq__(var_8)
    assert var_10 is False
    var_11 = var_2.__eq__(var_8)
    assert var_11 is False
    var_12 = None
    var_13 = var_9.__eq__(var_1)
    assert var_13 is False
    var_14 = var_9.messages(add_prefix=var_3)
    var_15 = var_9.__eq__(var_12)
    assert var_15 is False
    var_16 = var_9.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_9.get(var_8)
    var_18 = var_17.__bool__()
    assert var_18 is False
    var_19 = var_2.__hash__()
    assert var_19 == 4004646046084524075
    var_20 = var_16.__repr__()
    assert var_20 == 'KeysView(BaseError(text=ValuesView(ValidationError([Message(text=ValidationResult(value=None), code=\'custom\', index=[\':$o\\tOgsB?6u"N5u">/\'], position=\':$o\\tOgsB?6u"N5u">/\')])), code=\'custom\'))'
    var_17.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2365
    var_3 = -2365
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = False
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.Position(var_0, var_5, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is False
    assert var_7.char_index == -2365
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_7.__eq__(var_0)
    assert var_9 is False
    var_10 = var_1.__eq__(var_3)
    assert var_10 is False
    module_0.ParseError(code=var_2, messages=var_4)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = True
    var_4 = -1716.32
    var_5 = -2038
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.Position(var_4, var_3, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == pytest.approx(-1716.32, abs=0.01, rel=0.01)
    assert var_7.column_no is True
    assert var_7.char_index == -2038
    var_8 = module_0.ValidationResult(value=var_6, error=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == "Message(text=None, code='custom')"
    assert var_8.error is None
    var_9 = var_7.__eq__(var_0)
    assert var_9 is False
    var_10 = var_1.__eq__(var_1)
    assert var_10 is True
    var_11 = None
    var_12 = var_1.__eq__(var_11)
    assert var_12 is False
    var_13 = var_1.__eq__(var_0)
    assert var_13 is False
    var_14 = module_0.ValidationError(text=var_2)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_14.__eq__(var_0)
    assert var_15 is False
    var_16 = var_14.__eq__(var_11)
    assert var_16 is False
    var_17 = var_14.messages()
    var_18 = var_14.__eq__(var_14)
    assert var_18 is True
    var_19 = var_14.keys()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_19) == 1
    var_14.get(var_19)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = ':$o\tOgsB?6u"N5u">/'
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_2.__hash__()
    assert var_4 == 4004646046084524075
    var_5 = module_0.ValidationError(text=var_3, key=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5) == 1
    var_6 = var_5.values()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_6) == 1
    var_7 = var_6.__hash__()
    var_8 = var_3.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = None
    var_10 = module_0.BaseError(text=var_6, position=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_2.__eq__(var_9)
    assert var_11 is False
    var_12 = var_2.__eq__(var_9)
    assert var_12 is False
    var_13 = var_5.__str__()
    assert var_13 == '{\':$o\\tOgsB?6u"N5u">/\': ValidationResult(value=None)}'
    var_14 = var_5.messages()
    var_15 = var_6.__eq__(var_9)
    var_16 = var_5.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = module_0.ValidationResult(value=var_13)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value == '{\':$o\\tOgsB?6u"N5u">/\': ValidationResult(value=None)}'
    assert var_17.error is None
    var_16.keys()

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = False
    var_4 = -1716.32
    var_5 = -2038
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = module_0.Position(var_4, var_3, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == pytest.approx(-1716.32, abs=0.01, rel=0.01)
    assert var_7.column_no is False
    assert var_7.char_index == -2038
    var_8 = module_0.ValidationResult(value=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_8.value).__module__}.{type(var_8.value).__qualname__}' == 'typesystem.base.Position'
    assert var_8.error is None
    var_9 = var_7.__eq__(var_0)
    assert var_9 is False
    var_10 = var_1.__eq__(var_1)
    assert var_10 is True
    var_11 = None
    var_12 = module_0.Message(text=var_11, index=var_0, position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text is None
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = var_12.__eq__(var_11)
    assert var_13 is False
    var_14 = 'VE~Ih?\r8|h\n?fI'
    var_15 = module_0.Message(text=var_14, position=var_0, end_position=var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'VE~Ih?\r8|h\n?fI'
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_15.__eq__(var_1)
    assert var_16 is False
    var_17 = None
    module_0.ValidationError(position=var_17)

def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = -2365
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = False
    var_5 = -1716.32
    var_6 = -2038
    var_7 = var_1.__repr__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = module_0.Position(var_5, var_4, var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == pytest.approx(-1716.32, abs=0.01, rel=0.01)
    assert var_8.column_no is False
    assert var_8.char_index == -2038
    var_9 = module_0.ValidationResult(value=var_7, error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == "Message(text=None, code='custom')"
    assert var_9.error is None
    var_10 = var_8.__eq__(var_0)
    assert var_10 is False
    var_11 = var_1.__eq__(var_1)
    assert var_11 is True
    var_12 = None
    var_13 = var_1.__eq__(var_12)
    assert var_13 is False
    var_14 = var_1.__eq__(var_0)
    assert var_14 is False
    var_15 = module_0.ValidationError(text=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15) == 1
    var_16 = var_15.__len__()
    assert var_16 == 1
    var_17 = var_15.__str__()
    assert var_17 == "Message(text=None, code='custom')"
    var_18 = var_15.messages()
    var_19 = var_15.__eq__(var_0)
    assert var_19 is False
    var_20 = var_15.__repr__()
    assert var_20 == 'ValidationError(text="Message(text=None, code=\'custom\')", code=\'custom\')'
    var_21 = var_15.__len__()
    assert var_21 == 1
    var_22 = '"]utGQu7)ho:7'
    var_23 = [var_2]
    with pytest.raises(AssertionError):
        module_0.Message(text=var_22, key=var_10, index=var_23)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'test'
    assert var_3.code == 'custom'
    assert var_3.index == ['username']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1.__str__()
    assert var_4 == "Message(text='test', code='custom')"
    var_5 = 1
    var_6 = 2
    var_7 = None
    var_8 = None
    var_9 = [var_3, var_3, var_3]
    var_10 = module_0.BaseError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_10.__eq__(var_8)
    assert var_11 is False
    var_12 = var_11.__eq__(var_7)
    var_13 = var_12.__repr__()
    assert var_13 == 'NotImplemented'
    var_14 = module_0.Position(var_5, var_6, var_6)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no == 1
    assert var_14.column_no == 2
    assert var_14.char_index == 2
    var_15 = None
    var_16 = var_14.__eq__(var_15)
    assert var_16 is False
    var_17 = var_3.__eq__(var_15)
    assert var_17 is False
    var_18 = module_0.Message(text=var_0, position=var_14)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text == 'test'
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert f'{type(var_18.start_position).__module__}.{type(var_18.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_18.end_position).__module__}.{type(var_18.end_position).__qualname__}' == 'typesystem.base.Position'
    var_19 = module_0.Position(var_5, var_6, var_17)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no == 1
    assert var_19.column_no == 2
    assert var_19.char_index is False
    var_20 = 5
    var_21 = 6
    var_22 = module_0.Position(var_5, var_20, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert var_22.line_no == 1
    assert var_22.column_no == 5
    assert var_22.char_index == 6
    var_23 = module_0.Message(text=var_0, start_position=var_19, end_position=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'test'
    assert var_23.code == 'custom'
    assert var_23.index == []
    assert f'{type(var_23.start_position).__module__}.{type(var_23.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_23.end_position).__module__}.{type(var_23.end_position).__qualname__}' == 'typesystem.base.Position'
    var_24 = module_0.Message(text=var_0, code=var_4)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Message'
    assert var_24.text == 'test'
    assert var_24.code == "Message(text='test', code='custom')"
    assert var_24.index == []
    assert var_24.start_position is None
    assert var_24.end_position is None
    var_25 = var_1.__eq__(var_18)
    assert var_25 is False
    var_26 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.Message'
    assert var_26.text == 'test'
    assert var_26.code == 'username'
    assert var_26.index == []
    assert var_26.start_position is None
    assert var_26.end_position is None
    var_27 = hash(var_26)
    var_28 = hash(var_26)
    var_29 = module_0.Message(text=var_0, code=var_27)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Message'
    assert var_29.text == 'test'
    assert var_29.code == 5624976388823361444
    assert var_29.index == []
    assert var_29.start_position is None
    assert var_29.end_position is None
    var_30 = repr(var_29)
    var_31 = repr(var_10)
    var_32 = module_0.Position(var_5, var_6, var_5)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.base.Position'
    assert var_32.line_no == 1
    assert var_32.column_no == 2
    assert var_32.char_index == 1
    var_33 = module_0.Message(text=var_0, position=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.base.Message'
    assert var_33.text == 'test'
    assert var_33.code == 'custom'
    assert var_33.index == []
    assert f'{type(var_33.start_position).__module__}.{type(var_33.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_33.end_position).__module__}.{type(var_33.end_position).__qualname__}' == 'typesystem.base.Position'
    var_34 = repr(var_33)
    var_35 = module_0.Position(var_5, var_6, var_16)
    assert f'{type(var_35).__module__}.{type(var_35).__qualname__}' == 'typesystem.base.Position'
    assert var_35.line_no == 1
    assert var_35.column_no == 2
    assert var_35.char_index is False
    var_36 = module_0.Message(text=var_0, start_position=var_35, end_position=var_31)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.base.Message'
    assert var_36.text == 'test'
    assert var_36.code == 'custom'
    assert var_36.index == []
    assert f'{type(var_36.start_position).__module__}.{type(var_36.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_36.end_position == "BaseError([Message(text='test', code='custom', index=['username']), Message(text='test', code='custom', index=['username']), Message(text='test', code='custom', index=['username'])])"
    var_37 = repr(var_36)
    module_0.ParseError(messages=var_30)

def test_case_36():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'test'
    assert var_3.code == 'max_length'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = module_0.Message(text=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'test'
    assert var_4.code == 'custom'
    assert f'{type(var_4.index).__module__}.{type(var_4.index).__qualname__}' == 'builtins.list'
    assert len(var_4.index) == 1
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'users'
    var_6 = 3
    var_7 = [var_5, var_6, var_2]
    var_8 = module_0.Message(text=var_0, index=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'test'
    assert var_8.code == 'custom'
    assert var_8.index == ['users', 3, 'max_length']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = 1
    var_10 = 2
    var_11 = module_0.Position(var_9, var_10, var_6)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no == 1
    assert var_11.column_no == 2
    assert var_11.char_index == 3
    var_12 = module_0.Message(text=var_0, position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'test'
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert f'{type(var_12.start_position).__module__}.{type(var_12.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_12.end_position).__module__}.{type(var_12.end_position).__qualname__}' == 'typesystem.base.Position'
    var_13 = module_0.Position(var_9, var_10, var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 1
    assert var_13.column_no == 2
    assert var_13.char_index == 3
    var_14 = 6
    var_15 = module_0.Position(var_9, var_9, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no == 1
    assert var_15.column_no == 1
    assert var_15.char_index == 6
    var_16 = module_0.Message(text=var_0, start_position=var_13, end_position=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'test'
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert f'{type(var_16.start_position).__module__}.{type(var_16.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_16.end_position).__module__}.{type(var_16.end_position).__qualname__}' == 'typesystem.base.Position'
    var_17 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'test'
    assert var_17.code == 'max_length'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text == 'test'
    assert var_18.code == 'max_length'
    assert var_18.index == []
    assert var_18.start_position is None
    assert var_18.end_position is None
    var_19 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text == 'test'
    assert var_19.code == 'max_length'
    assert var_19.index == []
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'test'
    assert var_20.code == 'max_length'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = hash(var_19)
    var_22 = hash(var_20)
    var_23 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'test'
    assert var_23.code == 'max_length'
    assert var_23.index == []
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = repr(var_23)
    assert var_24 == "Message(text='test', code='max_length')"
    var_25 = [var_5, var_6, var_5]
    var_26 = module_0.Message(text=var_0, index=var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.Message'
    assert var_26.text == 'test'
    assert var_26.code == 'custom'
    assert var_26.index == ['users', 3, 'users']
    assert var_26.start_position is None
    assert var_26.end_position is None
    var_27 = repr(var_26)
    var_28 = module_0.Position(var_9, var_10, var_6)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Position'
    assert var_28.line_no == 1
    assert var_28.column_no == 2
    assert var_28.char_index == 3
    var_29 = module_0.Message(text=var_0, position=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.Message'
    assert var_29.text == 'test'
    assert var_29.code == 'custom'
    assert var_29.index == []
    assert f'{type(var_29.start_position).__module__}.{type(var_29.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_29.end_position).__module__}.{type(var_29.end_position).__qualname__}' == 'typesystem.base.Position'
    var_30 = repr(var_29)
    assert var_30 == "Message(text='test', code='custom', position=Position(line_no=1, column_no=2, char_index=3))"
    var_31 = module_0.Position(var_9, var_10, var_6)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.base.Position'
    assert var_31.line_no == 1
    assert var_31.column_no == 2
    assert var_31.char_index == 3
    var_32 = module_0.Position(var_9, var_22, var_14)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.base.Position'
    assert var_32.line_no == 1
    assert var_32.column_no == -6309270152214311808
    assert var_32.char_index == 6
    var_33 = module_0.Message(text=var_0, start_position=var_31, end_position=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.base.Message'
    assert var_33.text == 'test'
    assert var_33.code == 'custom'
    assert var_33.index == []
    assert f'{type(var_33.start_position).__module__}.{type(var_33.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_33.end_position).__module__}.{type(var_33.end_position).__qualname__}' == 'typesystem.base.Position'
    var_34 = repr(var_33)

def test_case_37():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 1
    var_3 = 2
    var_4 = module_0.Position(var_2, var_3, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 1
    assert var_4.column_no == 2
    assert var_4.char_index == 1
    var_5 = 'max_length'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_0, code=var_5, key=var_6, position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'test'
    assert var_7.code == 'max_length'
    assert var_7.index == ['username']
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'typesystem.base.Position'
    var_8 = 'users'
    var_9 = [var_8, var_2, var_6]
    var_10 = module_0.Message(text=var_0, code=var_5, index=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'test'
    assert var_10.code == 'max_length'
    assert var_10.index == ['users', 1, 'username']
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = module_0.Position(var_2, var_3, var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no == 1
    assert var_11.column_no == 2
    assert var_11.char_index == 1
    var_12 = 5
    var_13 = 6
    var_14 = module_0.Position(var_2, var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no == 1
    assert var_14.column_no == 5
    assert var_14.char_index == 6
    var_15 = module_0.Message(text=var_0, start_position=var_11, end_position=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'test'
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.Position'
    var_16 = 'test'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_16, position=var_4, start_position=var_11)

def test_case_38():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = module_0.Message(text=var_0, key=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'test'
    assert var_3.code == 'custom'
    assert var_3.index == ['test']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 1
    var_5 = 2
    var_6 = module_0.Position(var_4, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 1
    assert var_6.column_no == 2
    assert var_6.char_index == 2
    var_7 = None
    var_8 = var_3.__eq__(var_7)
    assert var_8 is False
    var_9 = module_0.Message(text=var_0, position=var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'test'
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.end_position).__module__}.{type(var_9.end_position).__qualname__}' == 'typesystem.base.Position'
    var_10 = module_0.Position(var_4, var_5, var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 2
    assert var_10.char_index is False
    var_11 = 5
    var_12 = 6
    var_13 = module_0.Position(var_4, var_11, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 1
    assert var_13.column_no == 5
    assert var_13.char_index == 6
    var_14 = module_0.Message(text=var_0, start_position=var_10, end_position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'test'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert f'{type(var_14.start_position).__module__}.{type(var_14.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'typesystem.base.Position'
    var_15 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'test'
    assert var_15.code == 'max_length'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_1.__eq__(var_9)
    assert var_16 is False
    var_17 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'test'
    assert var_17.code == 'max_length'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = hash(var_17)
    var_19 = hash(var_17)
    var_20 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'test'
    assert var_20.code == 'max_length'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = repr(var_20)
    assert var_21 == "Message(text='test', code='max_length')"
    var_22 = module_0.Message(text=var_0, index=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Message'
    assert var_22.text == 'test'
    assert var_22.code == 'custom'
    assert var_22.index == "Message(text='test', code='max_length')"
    assert var_22.start_position is None
    assert var_22.end_position is None
    var_23 = repr(var_22)
    var_24 = module_0.Position(var_4, var_5, var_4)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Position'
    assert var_24.line_no == 1
    assert var_24.column_no == 2
    assert var_24.char_index == 1
    var_25 = module_0.Message(text=var_0, position=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text == 'test'
    assert var_25.code == 'custom'
    assert var_25.index == []
    assert f'{type(var_25.start_position).__module__}.{type(var_25.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_25.end_position).__module__}.{type(var_25.end_position).__qualname__}' == 'typesystem.base.Position'
    var_26 = repr(var_25)
    var_27 = module_0.Position(var_4, var_5, var_21)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Position'
    assert var_27.line_no == 1
    assert var_27.column_no == 2
    assert var_27.char_index == "Message(text='test', code='max_length')"
    var_28 = module_0.Message(text=var_0, start_position=var_27, end_position=var_23)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Message'
    assert var_28.text == 'test'
    assert var_28.code == 'custom'
    assert var_28.index == []
    assert f'{type(var_28.start_position).__module__}.{type(var_28.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_28.end_position == 'Message(text=\'test\', code=\'custom\', index="Message(text=\'test\', code=\'max_length\')")'
    var_29 = repr(var_28)

def test_case_39():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = 'username'
    var_4 = module_0.Message(text=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'test'
    assert var_4.code == 'custom'
    assert var_4.index == ['username']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 1
    var_6 = 2
    var_7 = module_0.Position(var_5, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 2
    assert var_7.char_index == 2
    var_8 = None
    var_9 = var_7.__eq__(var_8)
    assert var_9 is False
    var_10 = var_4.__eq__(var_8)
    assert var_10 is False
    var_11 = module_0.Message(text=var_0, position=var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'test'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.Position'
    var_12 = module_0.Position(var_5, var_6, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == 1
    assert var_12.column_no == 2
    assert var_12.char_index is False
    var_13 = 5
    var_14 = 6
    var_15 = module_0.Position(var_5, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no == 1
    assert var_15.column_no == 5
    assert var_15.char_index == 6
    var_16 = module_0.Message(text=var_0, start_position=var_12, end_position=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'test'
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert f'{type(var_16.start_position).__module__}.{type(var_16.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_16.end_position).__module__}.{type(var_16.end_position).__qualname__}' == 'typesystem.base.Position'
    var_17 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'test'
    assert var_17.code == 'max_length'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text == 'test'
    assert var_18.code == 'max_length'
    assert var_18.index == []
    assert var_18.start_position is None
    assert var_18.end_position is None
    var_19 = var_18.__eq__(var_11)
    assert var_19 is False
    var_20 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'test'
    assert var_20.code == 'max_length'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = hash(var_20)
    var_22 = hash(var_20)
    var_23 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'test'
    assert var_23.code == 'max_length'
    assert var_23.index == []
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = repr(var_23)
    assert var_24 == "Message(text='test', code='max_length')"
    var_25 = module_0.Message(text=var_0, index=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text == 'test'
    assert var_25.code == 'custom'
    assert var_25.index == "Message(text='test', code='max_length')"
    assert var_25.start_position is None
    assert var_25.end_position is None
    var_26 = repr(var_25)
    var_27 = module_0.Position(var_5, var_6, var_5)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Position'
    assert var_27.line_no == 1
    assert var_27.column_no == 2
    assert var_27.char_index == 1
    var_28 = module_0.Message(text=var_0, position=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Message'
    assert var_28.text == 'test'
    assert var_28.code == 'custom'
    assert var_28.index == []
    assert f'{type(var_28.start_position).__module__}.{type(var_28.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_28.end_position).__module__}.{type(var_28.end_position).__qualname__}' == 'typesystem.base.Position'
    var_29 = repr(var_28)
    var_30 = module_0.Position(var_5, var_6, var_9)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.Position'
    assert var_30.line_no == 1
    assert var_30.column_no == 2
    assert var_30.char_index is False
    var_31 = module_0.Position(var_5, var_13, var_14)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.base.Position'
    assert var_31.line_no == 1
    assert var_31.column_no == 5
    assert var_31.char_index == 6
    var_32 = module_0.Message(text=var_0, start_position=var_30, end_position=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.base.Message'
    assert var_32.text == 'test'
    assert var_32.code == 'custom'
    assert var_32.index == []
    assert f'{type(var_32.start_position).__module__}.{type(var_32.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_32.end_position).__module__}.{type(var_32.end_position).__qualname__}' == 'typesystem.base.Position'
    var_33 = repr(var_32)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = 'username'
    var_4 = module_0.Message(text=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'test'
    assert var_4.code == 'custom'
    assert var_4.index == ['username']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 1
    var_6 = 0
    var_7 = None
    var_8 = module_0.Position(var_5, var_6, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no == 1
    assert var_8.column_no == 0
    assert var_8.char_index is None
    var_9 = None
    var_10 = var_4.__eq__(var_9)
    assert var_10 is False
    var_11 = module_0.Message(text=var_0, position=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'test'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.Position'
    var_12 = module_0.Position(var_5, var_6, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == 1
    assert var_12.column_no == 0
    assert var_12.char_index is False
    var_13 = 6
    var_14 = module_0.Message(text=var_0, start_position=var_12, end_position=var_12)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'test'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert f'{type(var_14.start_position).__module__}.{type(var_14.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'typesystem.base.Position'
    var_15 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'test'
    assert var_15.code == 'max_length'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_1.__eq__(var_11)
    assert var_16 is False
    var_17 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'test'
    assert var_17.code == 'max_length'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = hash(var_17)
    var_19 = hash(var_17)
    var_20 = module_0.Message(text=var_0, index=var_13)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'test'
    assert var_20.code == 'custom'
    assert var_20.index == 6
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = module_0.Position(var_5, var_6, var_5)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Position'
    assert var_21.line_no == 1
    assert var_21.column_no == 0
    assert var_21.char_index == 1
    var_22 = module_0.Message(text=var_0, position=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Message'
    assert var_22.text == 'test'
    assert var_22.code == 'custom'
    assert var_22.index == []
    assert f'{type(var_22.start_position).__module__}.{type(var_22.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_22.end_position).__module__}.{type(var_22.end_position).__qualname__}' == 'typesystem.base.Position'
    var_23 = repr(var_22)
    module_0.BaseError(key=var_9, position=var_7, messages=var_23)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'test'
    var_1 = module_0.Message(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'test'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'max_length'
    var_3 = 'user2ame'
    var_4 = module_0.Message(text=var_0, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'test'
    assert var_4.code == 'custom'
    assert var_4.index == ['user2ame']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 1
    var_6 = 2
    var_7 = module_0.Position(var_5, var_6, var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no == 1
    assert var_7.column_no == 2
    assert var_7.char_index == 2
    var_8 = None
    var_9 = var_7.__eq__(var_8)
    assert var_9 is False
    var_10 = module_0.Position(var_5, var_6, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 2
    assert var_10.char_index is False
    var_11 = 5
    var_12 = 6
    var_13 = module_0.Position(var_5, var_11, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no == 1
    assert var_13.column_no == 5
    assert var_13.char_index == 6
    var_14 = module_0.Message(text=var_0, start_position=var_10, end_position=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'test'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert f'{type(var_14.start_position).__module__}.{type(var_14.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'typesystem.base.Position'
    var_15 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'test'
    assert var_15.code == 'max_length'
    assert var_15.index == []
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = var_1.__eq__(var_4)
    assert var_16 is False
    var_17 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'test'
    assert var_17.code == 'max_length'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = hash(var_17)
    var_19 = hash(var_17)
    var_20 = module_0.Message(text=var_0, code=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'test'
    assert var_20.code == 'max_length'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = repr(var_20)
    assert var_21 == "Message(text='test', code='max_length')"
    var_22 = module_0.Message(text=var_0, index=var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Message'
    assert var_22.text == 'test'
    assert var_22.code == 'custom'
    assert var_22.index == "Message(text='test', code='max_length')"
    assert var_22.start_position is None
    assert var_22.end_position is None
    var_23 = module_0.Position(var_5, var_6, var_5)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Position'
    assert var_23.line_no == 1
    assert var_23.column_no == 2
    assert var_23.char_index == 1
    var_24 = module_0.Message(text=var_0, position=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Message'
    assert var_24.text == 'test'
    assert var_24.code == 'custom'
    assert var_24.index == []
    assert f'{type(var_24.start_position).__module__}.{type(var_24.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_24.end_position).__module__}.{type(var_24.end_position).__qualname__}' == 'typesystem.base.Position'
    var_25 = repr(var_24)
    var_2.keys()

def test_case_42():
    var_0 = 'Invalid input'
    var_1 = 'invalid'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = str(var_2)
    assert var_3 == 'Invalid input'
    var_4 = 'username'
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = str(var_5)
    assert var_6 == "{'username': 'Invalid input'}"
    var_7 = module_0.Message(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Invalid input'
    assert var_7.code == 'invalid'
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
    var_13 = module_0.BaseError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_13) == 2
    var_14 = str(var_13)
    assert var_14 == "{'username': 'Invalid input', 'password': 'Too short'}"
    var_15 = 'user'
    var_16 = [var_15, var_4]
    var_17 = module_0.Message(text=var_0, code=var_1, index=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'Invalid input'
    assert var_17.code == 'invalid'
    assert var_17.index == ['user', 'username']
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = [var_15, var_10]
    var_19 = module_0.Message(text=var_8, code=var_9, index=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text == 'Too short'
    assert var_19.code == 'min_length'
    assert var_19.index == ['user', 'password']
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = [var_17, var_19]
    var_21 = module_0.BaseError(messages=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_21) == 1
    var_22 = str(var_21)
    assert var_22 == "{'user': {'username': 'Invalid input', 'password': 'Too short'}}"
    var_23 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_23)