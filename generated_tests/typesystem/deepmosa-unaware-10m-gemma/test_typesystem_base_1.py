# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0
import builtins as module_1

def test_case_0():
    var_0 = '.xs'
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '.xs'
    assert var_1.code == 'custom'
    assert var_1.index == '.xs'
    assert var_1.start_position == '.xs'
    assert var_1.end_position == '.xs'

def test_case_1():
    var_0 = ''
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, code=var_0, key=var_0, index=var_0, position=var_0)

def test_case_2():
    with pytest.raises(AssertionError):
        module_0.BaseError()

@pytest.mark.xfail(strict=True)
def test_case_3():
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
    module_0.ParseError(messages=var_1)

def test_case_4():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False

def test_case_5():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "BaseError(text='.xs', code='custom')"

def test_case_6():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__contains__(var_0)
    assert var_2 is False
    var_3 = var_1.__str__()
    assert var_3 == '.xs'
    var_4 = var_2.__repr__()
    assert var_4 == 'False'

def test_case_7():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_8():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.Position(var_2, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert f'{type(var_3.char_index).__module__}.{type(var_3.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.char_index) == 1
    var_4 = None
    var_5 = ')[_\x0c\r^(\x0b/'
    var_6 = module_1.BaseException()
    var_7 = 'r\na'
    var_8 = module_0.Message(text=var_7, key=var_2, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'r\na'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == '.xs'
    assert var_8.end_position is None
    var_9 = module_0.ValidationResult(value=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == ')[_\x0c\r^(\x0b/'
    assert var_9.error is None
    var_10 = var_8.__eq__(var_9)
    assert var_10 is False
    var_11 = var_8.__repr__()
    assert var_11 == "Message(text='r\\na', code='custom', start_position='.xs', end_position=None)"
    var_12 = var_1.__contains__(var_4)
    assert var_12 is False
    var_13 = var_0.__iter__()
    var_14 = var_9.__repr__()
    assert var_14 == "ValidationResult(value=')[_\\x0c\\r^(\\x0b/')"

def test_case_9():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_10():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
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
    var_4 = 'eK=mb~z1=5'
    module_0.ValidationError(key=var_4, messages=var_4)

def test_case_12():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, index=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = False
    var_3 = True
    var_4 = module_0.Position(var_0, var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is False
    assert var_4.char_index is True
    var_5 = var_4.__eq__(var_0)
    assert var_5 is False

def test_case_13():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1
    var_4 = var_1.__repr__()
    assert var_4 == "BaseError(text='.xs', code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.messages()
    var_4 = var_1.__eq__(var_2)
    assert var_4 is False
    var_5 = var_1.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "KeysView(BaseError(text='.xs', code='custom'))"
    var_7 = var_1.keys()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7) == 1
    var_1.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.BaseError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_1.__contains__(var_1)
    assert var_3 is False
    var_4 = None
    var_5 = var_1.__iter__()
    var_6 = var_5.__eq__(var_4)
    var_5.keys()

def test_case_16():
    var_0 = ''
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0, messages=var_0)

def test_case_17():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.BaseError(text=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__iter__()
    var_5 = var_4.__repr__()
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_4, key=var_1, messages=var_4)

def test_case_18():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.messages(add_prefix=var_1)
    var_4 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__eq__(var_2)
    assert var_5 is False
    var_6 = var_4.__iter__()
    var_7 = var_5.__eq__(var_2)
    var_5.keys()

def test_case_20():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.messages()
    var_4 = var_1.__str__()
    assert var_4 == '.xs'
    var_5 = var_1.__contains__(var_2)
    assert var_5 is False
    var_6 = var_5.__bool__()
    assert var_6 is False
    var_7 = [var_0, var_1, var_6]
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_7, error=var_5)

def test_case_21():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '.xs'
    assert var_2.code == 'custom'
    assert var_2.index == ['.xs']
    assert var_2.start_position is None
    assert var_2.end_position is None
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_1)

def test_case_22():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.Position(var_2, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert f'{type(var_3.char_index).__module__}.{type(var_3.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.char_index) == 1
    var_4 = 'r\nP'
    var_5 = module_0.Message(text=var_4, key=var_2, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'r\nP'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position == '.xs'
    assert var_5.end_position is None
    var_6 = var_3.__repr__()
    assert var_6 == 'Position(line_no=None, column_no=None, char_index=.xs)'
    var_7 = var_3.__eq__(var_3)
    assert var_7 is False
    var_8 = var_5.__repr__()
    assert var_8 == "Message(text='r\\nP', code='custom', start_position='.xs', end_position=None)"
    var_9 = module_0.BaseError(text=var_6, code=var_2)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = {}
    module_0.ParseError(text=var_1, key=var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = var_4.__len__()
    assert var_7 == 1
    var_8 = var_7.__repr__()
    assert var_8 == '1'
    var_9 = var_4.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    module_0.ParseError(messages=var_9)

def test_case_25():
    var_0 = 'success_data'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success_data'
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = len(var_2)
    assert var_3 == 2
    var_4 = 'error text'
    var_5 = 'err_code'
    var_6 = module_0.ValidationError(text=var_4, code=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = module_0.ValidationResult(error=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = list(var_7)
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = list(var_2)
    var_12 = len(var_11)
    assert var_12 == 2

def test_case_26():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = '?m'
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '?m'
    assert var_1.code == 'custom'
    assert var_1.index == '?m'
    assert var_1.start_position == '?m'
    assert var_1.end_position == '?m'
    var_2 = module_0.BaseError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = var_1.__eq__(var_1)
    assert var_4 is True
    var_5 = var_2.__str__()
    assert var_5 == '?m'
    var_6 = var_1.__repr__()
    assert var_6 == "Message(text='?m', code='custom', index='?m', position='?m')"
    var_7 = 'eK=mb~z1=5'
    var_8 = var_2.__eq__(var_3)
    assert var_8 is False
    var_9 = var_8.__repr__()
    assert var_9 == 'False'
    module_0.ValidationError(key=var_7, messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = None
    var_6 = module_0.Message(text=var_5, key=var_5, position=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is False
    assert var_6.end_position is False
    var_7 = var_6.__eq__(var_0)
    assert var_7 is False
    var_8 = var_4.__iter__()
    var_9 = var_6.__repr__()
    assert var_9 == "Message(text=None, code='custom', position=False)"
    var_10 = var_4.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    var_11 = var_4.__len__()
    assert var_11 == 1
    var_7.__getitem__(var_8)

def test_case_29():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, index=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.items()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "ItemsView(BaseError(text='.xs', code='custom'))"
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_5, messages=var_5)

def test_case_30():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = module_0.Position(var_2, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert f'{type(var_3.char_index).__module__}.{type(var_3.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.char_index) == 1
    var_4 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_4.__bool__()
    assert var_5 is True
    var_6 = module_1.BaseException()
    var_7 = 'r\na'
    var_8 = module_0.Message(text=var_7, key=var_2, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'r\na'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == '.xs'
    assert var_8.end_position is None
    var_9 = var_1.__eq__(var_2)
    assert var_9 is False
    var_10 = var_8.__eq__(var_2)
    assert var_10 is False
    var_11 = var_8.__repr__()
    assert var_11 == "Message(text='r\\na', code='custom', start_position='.xs', end_position=None)"
    var_12 = var_1.keys()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_12) == 1
    var_13 = var_3.__eq__(var_12)
    assert var_13 is False
    var_14 = var_12.__str__()
    assert var_14 == "KeysView(BaseError(text='.xs', code='custom'))"
    var_15 = ()
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_2, position=var_3, messages=var_15)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = []
    module_0.ParseError(messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 3038025890804766121
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_0)
    var_6 = None
    var_7 = True
    var_8 = module_0.Position(var_6, var_6, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert f'{type(var_8.char_index).__module__}.{type(var_8.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8.char_index) == 1
    var_9 = module_1.BaseException()
    var_10 = 'r\nP'
    var_11 = module_0.Message(text=var_10, key=var_6, start_position=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'r\nP'
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position == '.xs'
    assert var_11.end_position is None
    var_12 = var_4.__eq__(var_6)
    assert var_12 is False
    var_13 = var_11.__eq__(var_6)
    assert var_13 is False
    var_14 = var_8.__repr__()
    assert var_14 == 'Position(line_no=None, column_no=None, char_index=.xs)'
    var_15 = var_11.__repr__()
    assert var_15 == "Message(text='r\\nP', code='custom', start_position='.xs', end_position=None)"
    var_16 = var_4.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = module_0.BaseError(text=var_7)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_17) == 1
    var_18 = var_4.__repr__()
    assert var_18 == "BaseError(text='.xs', code='custom')"
    var_19 = module_0.ValidationResult(error=var_16)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_19.value is None
    assert f'{type(var_19.error).__module__}.{type(var_19.error).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_19.error) == 1
    var_20 = var_19.__repr__()
    assert var_20 == "ValidationResult(error=KeysView(BaseError(text='.xs', code='custom')))"
    var_4.__getitem__(var_6)

def test_case_33():
    var_0 = '.xs'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages(add_prefix=var_0)
    var_3 = None
    var_4 = module_0.Position(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert f'{type(var_4.char_index).__module__}.{type(var_4.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4.char_index) == 1
    var_5 = None
    var_6 = var_1.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = 'r\nP'
    var_8 = module_0.Message(text=var_7, key=var_3, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'r\nP'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == '.xs'
    assert var_8.end_position is None
    var_9 = var_1.__eq__(var_3)
    assert var_9 is False
    var_10 = var_8.__eq__(var_3)
    assert var_10 is False
    var_11 = var_4.__repr__()
    assert var_11 == 'Position(line_no=None, column_no=None, char_index=.xs)'
    var_12 = var_8.__repr__()
    assert var_12 == "Message(text='r\\nP', code='custom', start_position='.xs', end_position=None)"
    var_13 = var_1.__str__()
    assert var_13 == '.xs'
    var_14 = module_0.BaseError(text=var_0, key=var_10, position=var_4)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14) == 1
    var_15 = var_14.__repr__()
    assert var_15 == "BaseError([Message(text='.xs', code='custom', index=[False], start_position=Position(line_no=None, column_no=None, char_index=.xs), end_position=Position(line_no=None, column_no=None, char_index=.xs))])"
    var_16 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_9.__repr__()
    assert var_17 == 'False'

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_0)
    var_6 = None
    var_7 = module_0.Position(var_6, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is None
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7.char_index) == 1
    var_8 = module_1.BaseException()
    var_9 = 'r\nP'
    var_10 = module_0.Message(text=var_9, key=var_6, start_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'r\nP'
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position == '.xs'
    assert var_10.end_position is None
    var_11 = var_4.__eq__(var_8)
    assert var_11 is False
    var_12 = var_10.__eq__(var_2)
    assert var_12 is False
    var_13 = var_7.__repr__()
    assert var_13 == 'Position(line_no=None, column_no=None, char_index=.xs)'
    var_14 = var_2.__repr__()
    assert var_14 == "Message(text=None, code='custom')"
    var_15 = var_4.keys()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_15) == 1
    var_16 = var_7.__eq__(var_15)
    assert var_16 is False
    var_4.__getitem__(var_1)

def test_case_35():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_1)
    var_6 = None
    var_7 = module_0.Position(var_3, var_6, var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is None
    var_8 = var_4.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = '[t'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_9, position=var_7, end_position=var_7)

def test_case_36():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_0)
    var_6 = None
    var_7 = module_0.Position(var_6, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is None
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7.char_index) == 1
    var_8 = None
    var_9 = module_1.BaseException()
    var_10 = var_4.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_6, key=var_8, position=var_7, start_position=var_10)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = 'Simple error'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Simple error'
    assert var_2.code == 'err_code'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 4481962520575761495
    var_4 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = str(var_4)
    assert var_5 == 'Simple error'
    var_6 = 'Jsm^M/e],Apz+4>&'
    var_7 = 'username'
    var_8 = module_0.Message(text=var_6, key=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Jsm^M/e],Apz+4>&'
    assert var_8.code == 'custom'
    assert var_8.index == ['username']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = [var_8]
    var_10 = module_0.ValidationError(messages=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10) == 1
    var_11 = var_8.__eq__(var_5)
    assert var_11 is False
    var_12 = str(var_10)
    var_13 = 'Error A'
    var_14 = 'users'
    var_15 = var_10.__eq__(var_12)
    assert var_15 is False
    var_16 = 1
    var_17 = [var_14, var_16, var_7]
    var_18 = var_12.__hash__()
    assert var_18 == 2313011487729204587
    var_19 = module_0.Message(text=var_13, index=var_17)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text == 'Error A'
    assert var_19.code == 'custom'
    assert var_19.index == ['users', 1, 'username']
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_15.__len__()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = '`.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_0)
    var_6 = None
    var_7 = var_4.__iter__()
    var_8 = module_0.Position(var_6, var_6, var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert f'{type(var_8.char_index).__module__}.{type(var_8.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8.char_index) == 1
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_9.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_4.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = 'r\nP'
    var_13 = module_0.Message(text=var_12, key=var_6, start_position=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == 'r\nP'
    assert var_13.code == 'custom'
    assert var_13.index == []
    assert var_13.start_position == '`.xs'
    assert var_13.end_position is None
    var_14 = var_4.__eq__(var_6)
    assert var_14 is False
    var_15 = var_13.__eq__(var_6)
    assert var_15 is False
    var_16 = var_8.__repr__()
    assert var_16 == 'Position(line_no=None, column_no=None, char_index=`.xs)'
    var_17 = var_13.__repr__()
    assert var_17 == "Message(text='r\\nP', code='custom', start_position='`.xs', end_position=None)"
    var_18 = var_2.__repr__()
    assert var_18 == "Message(text=None, code='custom')"
    var_19 = module_0.Position(var_15, var_11, var_7)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no is False
    assert f'{type(var_19.column_no).__module__}.{type(var_19.column_no).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_19.column_no) == 1
    assert f'{type(var_19.char_index).__module__}.{type(var_19.char_index).__qualname__}' == 'builtins.dict_keyiterator'
    var_20 = var_8.__eq__(var_19)
    assert var_20 is False
    var_21 = var_4.__str__()
    assert var_21 == '`.xs'
    var_22 = module_0.BaseError(text=var_19, key=var_15, position=var_8)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_22) == 1
    var_23 = var_22.__repr__()
    var_24 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_24.value is None
    assert var_24.error is None
    var_25 = module_0.ValidationResult(value=var_5)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_25.value).__module__}.{type(var_25.value).__qualname__}' == 'builtins.list'
    assert len(var_25.value) == 1
    assert var_25.error is None
    var_26 = var_25.__repr__()
    assert var_26 == "ValidationResult(value=[Message(text='`.xs', code='custom', index=['`.xs'])])"
    var_22.__getitem__(var_2)

def test_case_39():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_0)
    var_6 = None
    var_7 = True
    var_8 = None
    var_9 = var_4.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = module_0.Message(text=var_8, position=var_6, start_position=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert f'{type(var_10.start_position).__module__}.{type(var_10.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10.start_position) == 1
    assert var_10.end_position is None
    var_11 = var_4.__eq__(var_6)
    assert var_11 is False
    var_12 = var_2.__eq__(var_10)
    assert var_12 is False
    var_13 = var_9.__repr__()
    assert var_13 == "ItemsView(BaseError(text='.xs', code='custom'))"
    var_14 = var_10.__repr__()
    assert var_14 == "Message(text=None, code='custom', start_position=ItemsView(BaseError(text='.xs', code='custom')), end_position=None)"
    var_15 = var_13.__repr__()
    assert var_15 == '"ItemsView(BaseError(text=\'.xs\', code=\'custom\'))"'
    var_16 = var_4.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    var_17 = var_7.__eq__(var_7)
    assert var_17 is True
    var_18 = var_16.__str__()
    assert var_18 == "KeysView(BaseError(text='.xs', code='custom'))"
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_40():
    var_0 = '`.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_0, code=var_1, position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '`.xs'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = [var_2, var_2, var_2]
    var_5 = module_0.BaseError(position=var_1, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = None
    var_8 = False
    var_9 = module_0.Position(var_8, var_7, var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is False
    assert var_9.column_no is None
    assert var_9.char_index is None
    var_10 = None
    var_11 = '|E+MX4&:@Kl[Wj'
    var_12 = module_1.BaseException()
    var_13 = module_0.Message(text=var_1, key=var_11, start_position=var_10, end_position=var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text is None
    assert var_13.code == 'custom'
    assert var_13.index == ['|E+MX4&:@Kl[Wj']
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = var_5.__eq__(var_1)
    assert var_14 is False
    var_15 = var_2.__eq__(var_1)
    assert var_15 is False
    var_16 = var_9.__repr__()
    assert var_16 == 'Position(line_no=False, column_no=None, char_index=None)'
    var_17 = var_13.__repr__()
    assert var_17 == "Message(text=None, code='custom', index=['|E+MX4&:@Kl[Wj'])"
    var_18 = var_13.__repr__()
    assert var_18 == "Message(text=None, code='custom', index=['|E+MX4&:@Kl[Wj'])"
    var_19 = var_5.keys()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_19) == 1
    var_20 = var_19.__eq__(var_5)
    var_21 = var_19.__str__()
    assert var_21 == "KeysView(BaseError([Message(text='`.xs', code='custom'), Message(text='`.xs', code='custom'), Message(text='`.xs', code='custom')]))"
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_1, key=var_16)

def test_case_41():
    var_0 = 'Simple error'
    var_1 = 'err_code'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Simple error'
    assert var_2.code == 'err_code'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = str(var_3)
    assert var_4 == 'Simple error'
    var_5 = 'Field error'
    var_6 = 'username'
    var_7 = module_0.Message(text=var_5, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Field error'
    assert var_7.code == 'custom'
    assert var_7.index == ['username']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_7]
    var_9 = module_0.ValidationError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = 'Error A'
    var_11 = 'users'
    var_12 = 0
    var_13 = 'name'
    var_14 = [var_11, var_12, var_13]
    var_15 = module_0.Message(text=var_10, index=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'Error A'
    assert var_15.code == 'custom'
    assert var_15.index == ['users', 0, 'name']
    assert var_15.start_position is None
    assert var_15.end_position is None
    var_16 = 'Error B'
    var_17 = 1
    var_18 = 'age'
    var_19 = [var_11, var_17, var_18]
    var_20 = module_0.Message(text=var_16, index=var_19)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'Error B'
    assert var_20.code == 'custom'
    assert var_20.index == ['users', 1, 'age']
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = 'Error C'
    var_22 = 'global_key'
    var_23 = module_0.Message(text=var_21, key=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'Error C'
    assert var_23.code == 'custom'
    assert var_23.index == ['global_key']
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = [var_15, var_20, var_23]
    var_25 = module_0.ValidationError(messages=var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_25) == 2
    var_26 = {var_13: var_10}
    var_27 = {var_18: var_16}
    var_28 = {var_12: var_26, var_17: var_27}
    var_29 = {var_11: var_28, var_22: var_21}
    var_30 = str(var_29)
    var_31 = str(var_25)
    var_32 = 'Root error'
    var_33 = []
    var_34 = module_0.Message(text=var_32, index=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.base.Message'
    assert var_34.text == 'Root error'
    assert var_34.code == 'custom'
    assert var_34.index == []
    assert var_34.start_position is None
    assert var_34.end_position is None
    var_35 = [var_34]
    var_36 = module_0.ValidationError(messages=var_35)
    assert f'{type(var_36).__module__}.{type(var_36).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_36) == 1
    var_37 = str(var_36)

def test_case_42():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 4
    assert var_3.column_no == 5
    assert var_3.char_index == 6
    var_4 = 'eR]r\x0cor 1'
    var_5 = 'err1'
    var_6 = 'a'
    var_7 = 0
    var_8 = [var_6, var_7]
    var_9 = module_0.Message(text=var_4, code=var_5, index=var_8, position=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'eR]r\x0cor 1'
    assert var_9.code == 'err1'
    assert var_9.index == ['a', 0]
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.end_position).__module__}.{type(var_9.end_position).__qualname__}' == 'typesystem.base.Position'
    var_10 = [var_6, var_7]
    var_11 = module_0.Message(text=var_4, code=var_5, index=var_10, position=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'eR]r\x0cor 1'
    assert var_11.code == 'err1'
    assert var_11.index == ['a', 0]
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.Position'
    var_12 = 'err2'
    var_13 = 'b'
    var_14 = [var_13]
    var_15 = module_0.Message(text=var_12, code=var_12, index=var_14, position=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 'err2'
    assert var_15.code == 'err2'
    assert var_15.index == ['b']
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.Position'
    var_16 = [var_9, var_15]
    var_17 = module_0.ValidationError(messages=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17) == 2
    var_18 = [var_9, var_15]
    var_19 = module_0.ValidationError(messages=var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_19) == 2
    var_20 = [var_15]
    var_21 = module_0.ValidationError(messages=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_21) == 1

def test_case_43():
    var_0 = '.xs'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '.xs'
    assert var_1.code == '.xs'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = module_0.Message(text=var_0, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '.xs'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_44():
    var_0 = '`.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages()
    var_6 = None
    var_7 = var_2.__hash__()
    assert var_7 == 3038025890804766121
    var_8 = False
    var_9 = module_0.Position(var_6, var_1, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no is None
    assert var_9.char_index is False
    var_10 = None
    var_11 = var_4.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = module_0.Message(text=var_6, start_position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text is None
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert f'{type(var_12.start_position).__module__}.{type(var_12.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12.start_position) == 1
    assert var_12.end_position is None
    var_13 = '@SF(T,'
    var_14 = module_0.ValidationError(text=var_13, code=var_1, key=var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_4.__eq__(var_14)
    assert var_15 is False
    var_16 = var_2.__eq__(var_9)
    assert var_16 is False
    var_17 = var_11.__repr__()
    assert var_17 == "ItemsView(BaseError(text='`.xs', code='custom'))"
    var_18 = var_11.__len__()
    assert var_18 == 1
    var_19 = var_18.__repr__()
    assert var_19 == '1'
    var_20 = var_12.__repr__()
    assert var_20 == "Message(text=None, code='custom', start_position=ItemsView(BaseError(text='`.xs', code='custom')), end_position=None)"
    var_21 = var_14.keys()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_21) == 1
    var_22 = var_21.__eq__(var_21)
    assert var_22 is True
    var_23 = var_14.__str__()
    assert var_23 == '@SF(T,'
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_21, position=var_10)

def test_case_45():
    var_0 = '.xs'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '.xs'
    assert var_1.code == '.xs'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = module_0.BaseError(text=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.messages(add_prefix=var_0)
    var_5 = None
    var_6 = module_0.Position(var_5, var_5, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no is None
    assert var_6.column_no is None
    assert f'{type(var_6.char_index).__module__}.{type(var_6.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6.char_index) == 1
    var_7 = None
    var_8 = var_3.__eq__(var_5)
    assert var_8 is False
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_9.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = var_3.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = "dy,?#)'-(:R=8je~c"
    var_13 = module_0.Message(text=var_12, position=var_7, end_position=var_7)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text == "dy,?#)'-(:R=8je~c"
    assert var_13.code == 'custom'
    assert var_13.index == []
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = var_3.__eq__(var_5)
    assert var_14 is False
    var_15 = var_1.__eq__(var_7)
    assert var_15 is False
    var_16 = var_6.__repr__()
    assert var_16 == 'Position(line_no=None, column_no=None, char_index=.xs)'
    var_17 = var_11.__repr__()
    assert var_17 == "ItemsView(BaseError(text='.xs', code='custom'))"
    var_18 = var_1.__repr__()
    assert var_18 == "Message(text='.xs', code='.xs')"
    var_19 = module_0.Position(var_5, var_14, var_7)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no is None
    assert var_19.column_no is False
    assert var_19.char_index is None
    var_20 = var_19.__eq__(var_6)
    assert var_20 is False
    var_21 = var_3.__contains__(var_7)
    assert var_21 is False
    var_22 = var_21.__eq__(var_7)
    var_23 = var_11.__str__()
    assert var_23 == "ItemsView(BaseError(text='.xs', code='custom'))"
    var_24 = "B1k)YP\x0b'og,^8i.\\"
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_24, position=var_11, messages=var_21)

def test_case_46():
    var_0 = '.xs'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.BaseError(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.messages(add_prefix=var_0)
    var_6 = None
    var_7 = module_0.Position(var_6, var_6, var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is None
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7.char_index) == 1
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_8.__repr__()
    assert var_9 == 'ValidationResult(value=None)'
    var_10 = var_4.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    var_11 = module_0.Message(text=var_1, key=var_3, position=var_1)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is None
    assert var_11.code == 'custom'
    assert var_11.index == [True]
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_4.__eq__(var_6)
    assert var_12 is False
    var_13 = var_11.__eq__(var_2)
    assert var_13 is False
    var_14 = var_10.__repr__()
    assert var_14 == "ItemsView(BaseError(text='.xs', code='custom'))"
    var_15 = var_4.__eq__(var_10)
    assert var_15 is False
    var_16 = var_15.__eq__(var_10)
    var_17 = var_16.__repr__()
    assert var_17 == 'NotImplemented'
    var_18 = var_4.get(var_1)
    var_19 = var_18.__repr__()
    assert var_19 == 'None'
    var_20 = False
    var_21 = module_0.Position(var_1, var_20, var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Position'
    assert var_21.line_no is None
    assert var_21.column_no is False
    assert var_21.char_index is False
    var_22 = var_7.__eq__(var_1)
    assert var_22 is False
    var_23 = var_18.__eq__(var_1)
    assert var_23 is True
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_14, key=var_6)