# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = '\r2ZJF5BSP/k]Jz4)p\x0bE'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_1():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_2():
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_3():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_4():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_5():
    var_0 = '=OQ/Bj:b'
    var_1 = None
    var_2 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False

def test_case_6():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'code1'
    assert var_4.index == [0]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = module_0.Message(text=var_5, code=var_6, index=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error 2'
    assert var_7.code == 'code2'
    assert var_7.index == [0]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_4, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = str(var_9)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = True
    var_2 = module_0.ValidationError(text=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_2.__getitem__(var_0)

def test_case_8():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = module_0.Message(text=var_3, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = None
    var_6 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_6.__bool__()
    assert var_7 is True
    var_8 = var_4.__eq__(var_5)
    assert var_8 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_5)

def test_case_9():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_3, position=var_0)

def test_case_10():
    var_0 = 'Err1'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "ValidationError(text='Err1', code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = -1794
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == -1794
    assert var_2.char_index == -1794
    var_3 = False
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=None, column_no=-1794, char_index=-1794)'
    var_5 = module_0.Position(var_0, var_0, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is None
    assert var_5.column_no is None
    assert var_5.char_index is False
    module_0.ValidationError(position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = None
    var_3 = module_0.BaseError(text=var_1, key=var_2, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__len__()
    assert var_4 == 1
    var_4.__contains__(var_0)

def test_case_13():
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

def test_case_14():
    var_0 = True
    var_1 = module_0.ValidationError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_2) == 1

def test_case_15():
    var_0 = 'error_code'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = bool(var_1 == var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = var_2.__contains__(var_3)
    assert var_4 is False
    var_5 = var_4.__repr__()
    assert var_5 == 'False'
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = False
    var_8 = module_0.Position(var_4, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is False
    assert var_8.column_no is None
    assert var_8.char_index is False
    var_9 = var_6.__iter__()
    var_10 = var_2.__hash__()
    assert var_10 == 8866104294228179753
    var_11 = var_6.__bool__()
    assert var_11 is True
    var_12 = var_10.__eq__(var_10)
    assert var_12 is True
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_4.__len__()

def test_case_17():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = None
    var_4 = module_0.Message(text=var_3, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_1.__hash__()
    assert var_5 == 1371871601451571618
    var_6 = None
    var_7 = module_0.ValidationResult(error=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_7.__bool__()
    assert var_8 is True
    var_9 = var_4.__eq__(var_6)
    assert var_9 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_6)

def test_case_18():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__eq__(var_0)
    assert var_3 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_2)

def test_case_19():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'code1'
    assert var_4.index == ['field1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'code2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'code2'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = repr(var_11)
    var_13 = bool(var_12 == f'BaseError({var_10!r})')
    assert var_13 is True

def test_case_20():
    var_0 = 'd<ic*p#b=5n'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_0, messages=var_0)

def test_case_21():
    var_0 = None
    var_1 = 'KeYuB\necU\x0bX-M'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_0, key=var_1, index=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == ['KeYuB\necU\x0bX-M']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.__repr__()
    assert var_4 == "BaseError(text='KeYuB\\necU\\x0bX-M', code='custom')"
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = var_2.messages()
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, key=var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = 'd<ic*p#b=5\rn'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_4 = var_3.__contains__(var_0)
    assert var_4 is False
    var_5 = var_4.__str__()
    assert var_5 == 'False'
    var_6 = None
    var_7 = var_3.__repr__()
    assert var_7 == "ValuesView(BaseError(text='d<ic*p#b=5\\rn', code='custom'))"
    module_0.ParseError(text=var_6, key=var_4, messages=var_2)

def test_case_23():
    var_0 = 'Err\x0br1'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'Err\x0br1'
    assert var_1.code == 'Err\x0br1'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_24():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = iter(var_1)
    var_3 = next(var_2)
    var_4 = bool(var_3 == var_0)
    assert var_4 is True
    var_5 = next(var_2)
    assert var_5 is None

def test_case_25():
    var_0 = '2gatVrqCJ%h^'
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == '2gatVrqCJ%h^'
    assert var_1.code == 'custom'
    assert var_1.index == ['2gatVrqCJ%h^']
    assert var_1.start_position == '2gatVrqCJ%h^'
    assert var_1.end_position == '2gatVrqCJ%h^'
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_26():
    var_0 = None
    var_1 = 'd<icp#=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_2, code=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_3.text).__module__}.{type(var_3.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.text) == 1
    assert f'{type(var_3.code).__module__}.{type(var_3.code).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.code) == 1
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.start_position) == 1
    assert var_3.end_position is None
    var_4 = var_2.__str__()
    assert var_4 == 'd<icp#=5n'
    var_5 = var_2.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_5.__eq__(var_0)
    var_8 = var_5.__eq__(var_0)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_5, error=var_5)

def test_case_27():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = '5l;A0S_aS'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_4, code=var_0, key=var_3, position=var_3, start_position=var_1)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.messages(add_prefix=var_1)
    var_4 = '<K)'
    var_5 = module_0.Message(text=var_3, code=var_4, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'builtins.list'
    assert len(var_5.text) == 1
    assert var_5.code == '<K)'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text=[Message(text='d<ic*p#b=5n', code='custom', index=['d<ic*p#b=5n'])], code='<K)')"
    var_7 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value == 'd<ic*p#b=5n'
    assert var_7.error is None
    var_8 = var_5.__eq__(var_7)
    assert var_8 is False
    var_9 = var_5.__eq__(var_3)
    assert var_9 is False
    var_10 = var_2.__contains__(var_0)
    assert var_10 is False
    var_11 = var_10.__repr__()
    assert var_11 == 'False'
    var_12 = None
    module_0.BaseError(text=var_12, code=var_12, messages=var_10)

def test_case_29():
    var_0 = 'Error 1'
    var_1 = 33
    var_2 = [var_1, var_1, var_1]
    var_3 = module_0.Message(text=var_0, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error 1'
    assert var_3.code == 'Error 1'
    assert var_3.index == [33, 33, 33]
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'Error 2'
    var_5 = 'code2'
    var_6 = module_0.Message(text=var_4, code=var_5, index=var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 2'
    assert var_6.code == 'code2'
    assert var_6.index == [33, 33, 33]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = [var_3, var_6]
    var_8 = module_0.BaseError(messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = str(var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = 'd<icp#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_3, code=var_1, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_4.text).__module__}.{type(var_4.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4.text) == 1
    assert var_4.code == 'd<icp#b=5n'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = module_0.Message(text=var_3, key=var_3, index=var_0, start_position=var_4, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5.text) == 1
    assert var_5.code == 'custom'
    assert f'{type(var_5.index).__module__}.{type(var_5.index).__qualname__}' == 'builtins.list'
    assert len(var_5.index) == 1
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Message'
    assert var_5.end_position is None
    var_6 = var_3.__iter__()
    var_7 = var_4.__eq__(var_5)
    assert var_7 is False
    var_3.get(var_0)

def test_case_31():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationResult(value=var_0, error=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error == 'd<ic*p#b=5n'
    var_4 = module_0.Message(text=var_3, code=var_3, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_4.text).__module__}.{type(var_4.text).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.code).__module__}.{type(var_4.code).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=ValidationResult(error='d<ic*p#b=5n'), code=ValidationResult(error='d<ic*p#b=5n'))"
    var_6 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value == 'd<ic*p#b=5n'
    assert var_6.error is None
    var_7 = None
    var_8 = var_4.__eq__(var_6)
    assert var_8 is False
    var_9 = var_2.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.ValidationResult(value=var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_9.__eq__(var_0)
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_32():
    var_0 = None
    var_1 = 'd<ic\x0bp#=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_2, code=var_2, start_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_3.text).__module__}.{type(var_3.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.text) == 1
    assert f'{type(var_3.code).__module__}.{type(var_3.code).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.code) == 1
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.start_position) == 1
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text=BaseError(text='d<ic\\x0bp#=5n', code='custom'), code=BaseError(text='d<ic\\x0bp#=5n', code='custom'), start_position=BaseError(text='d<ic\\x0bp#=5n', code='custom'), end_position=None)"
    var_5 = var_2.__str__()
    assert var_5 == 'd<ic\x0bp#=5n'
    var_6 = var_3.__eq__(var_3)
    assert var_6 is False
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text=BaseError(text='d<ic\\x0bp#=5n', code='custom'), code=BaseError(text='d<ic\\x0bp#=5n', code='custom'), start_position=BaseError(text='d<ic\\x0bp#=5n', code='custom'), end_position=None)"
    var_8 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_7.__eq__(var_0)
    var_10 = var_7.__eq__(var_0)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_6, error=var_7)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = 'd<icp$b\x0c5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = '<K)'
    var_5 = module_0.Message(text=var_3, code=var_4, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5.text) == 1
    assert var_5.code == '<K)'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = module_0.Message(text=var_3, key=var_3, index=var_0, start_position=var_5, end_position=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_6.text).__module__}.{type(var_6.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6.text) == 1
    assert var_6.code == 'custom'
    assert f'{type(var_6.index).__module__}.{type(var_6.index).__qualname__}' == 'builtins.list'
    assert len(var_6.index) == 1
    assert f'{type(var_6.start_position).__module__}.{type(var_6.start_position).__qualname__}' == 'typesystem.base.Message'
    assert var_6.end_position is None
    var_7 = var_3.__iter__()
    var_8 = None
    var_9 = 'nLM\x0bq~(CsGkc'
    var_10 = module_0.Message(text=var_9, code=var_3, index=var_3, position=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'nLM\x0bq~(CsGkc'
    assert f'{type(var_10.code).__module__}.{type(var_10.code).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10.code) == 1
    assert f'{type(var_10.index).__module__}.{type(var_10.index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10.index) == 1
    assert f'{type(var_10.start_position).__module__}.{type(var_10.start_position).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_10.end_position).__module__}.{type(var_10.end_position).__qualname__}' == 'typesystem.base.Message'
    var_11 = var_10.__repr__()
    assert var_11 == "Message(text='nLM\\x0bq~(CsGkc', code=KeysView(BaseError(text='d<icp$b\\x0c5n', code='custom')), index=KeysView(BaseError(text='d<icp$b\\x0c5n', code='custom')), position=Message(text=KeysView(BaseError(text='d<icp$b\\x0c5n', code='custom')), code='custom', index=[KeysView(BaseError(text='d<icp$b\\x0c5n', code='custom'))], start_position=Message(text=KeysView(BaseError(text='d<icp$b\\x0c5n', code='custom')), code='<K)'), end_position=None))"
    var_12 = None
    var_13 = var_2.__eq__(var_0)
    assert var_13 is False
    var_14 = module_0.ValidationResult(value=var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    module_0.ParseError(code=var_12, position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = None
    var_1 = -1794
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == -1794
    assert var_2.char_index == -1794
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = True
    var_5 = module_0.Position(var_0, var_0, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is None
    assert var_5.column_no is None
    assert var_5.char_index is True
    module_0.ValidationError(position=var_5, messages=var_5)

def test_case_35():
    var_0 = None
    var_1 = 'd<icp#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'd<icp#b=5n'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_0)
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text='d<icp#b=5n', code='custom')"
    var_7 = set()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_7, key=var_4, index=var_4, start_position=var_2)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = 'd<icp#b=5n'
    var_2 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = '<K)'
    var_5 = module_0.Message(text=var_3, code=var_4, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.text) == 1
    assert var_5.code == '<K)'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text=BaseError(text='d<icp#b=5n', code='custom'), code='<K)')"
    var_7 = module_0.Message(text=var_4, key=var_4, index=var_0, start_position=var_5, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '<K)'
    assert var_7.code == 'custom'
    assert var_7.index == ['<K)']
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Message'
    assert var_7.end_position is None
    var_8 = var_4.__iter__()
    var_9 = None
    var_10 = var_5.__eq__(var_7)
    assert var_10 is False
    var_11 = None
    var_12 = var_3.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationResult(value=var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    module_0.ParseError(code=var_11, position=var_0)

def test_case_37():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, position=var_3, end_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = None
    var_1 = 'd<ic*p#b=5n'
    var_2 = module_0.BaseError(text=var_1, code=var_0, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_0, code=var_0, position=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position == 'd<ic*p#b=5n'
    assert var_4.end_position == 'd<ic*p#b=5n'
    var_5 = var_3.__repr__()
    assert var_5 == "KeysView(BaseError([Message(text='d<ic*p#b=5n', code='custom', index=['d<ic*p#b=5n'])]))"
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_2.get(var_3, var_0)

def test_case_39():
    var_0 = None
    var_1 = 'd<icp#b\x0c5n'
    var_2 = module_0.BaseError(text=var_1, key=var_0, messages=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = '<K)'
    var_5 = module_0.Message(text=var_3, code=var_4, start_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_5.text).__module__}.{type(var_5.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5.text) == 1
    assert var_5.code == '<K)'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text=KeysView(BaseError(text='d<icp#b\\x0c5n', code='custom')), code='<K)')"
    var_7 = module_0.Message(text=var_3, key=var_3, index=var_0, start_position=var_5, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_7.text).__module__}.{type(var_7.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7.text) == 1
    assert var_7.code == 'custom'
    assert f'{type(var_7.index).__module__}.{type(var_7.index).__qualname__}' == 'builtins.list'
    assert len(var_7.index) == 1
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'typesystem.base.Message'
    assert var_7.end_position is None
    var_8 = var_3.__iter__()
    var_9 = var_4.__eq__(var_0)
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = None
    var_12 = var_2.__contains__(var_0)
    assert var_12 is False
    var_13 = module_0.ParseError(text=var_3)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14) == 1
    var_15 = var_14.__eq__(var_0)
    var_16 = module_0.ValidationResult(value=var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16.value) == 1
    assert var_16.error is None
    var_17 = module_0.ValidationResult()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert var_17.error is None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_9, key=var_11, messages=var_17)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = None
    var_1 = []
    module_0.ValidationError(code=var_0, key=var_0, messages=var_1)

def test_case_41():
    var_0 = False
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is False
    assert var_1.column_no is False
    assert var_1.char_index is False
    var_2 = 'd<ic*p#b=5n'
    var_3 = -44
    var_4 = module_0.Position(var_2, var_0, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == 'd<ic*p#b=5n'
    assert var_4.column_no is False
    assert var_4.char_index == -44
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = var_4.__eq__(var_4)
    assert var_6 is True

def test_case_42():
    var_0 = None
    var_1 = 2197
    var_2 = True
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 2197
    assert var_3.column_no == 2197
    assert var_3.char_index is True
    var_4 = 720
    var_5 = module_0.Position(var_0, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is None
    assert var_5.column_no == 720
    assert var_5.char_index == 720
    var_6 = var_3.__eq__(var_5)
    assert var_6 is False
    var_7 = var_3.__eq__(var_0)
    assert var_7 is False
    var_8 = var_3.__eq__(var_0)
    assert var_8 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = None
    var_1 = 2197
    var_2 = True
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 2197
    assert var_3.column_no == 2197
    assert var_3.char_index is True
    var_4 = 'd<ic*p#b=5n'
    var_5 = False
    var_6 = module_0.Position(var_3, var_5, var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'typesystem.base.Position'
    assert var_6.column_no is False
    assert var_6.char_index is None
    var_7 = module_0.Position(var_5, var_0, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is False
    assert var_7.column_no is None
    assert var_7.char_index is None
    var_8 = var_7.__eq__(var_1)
    assert var_8 is False
    var_9 = var_6.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.BaseError(text=var_4, code=var_0, key=var_4, messages=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_10) == 1
    var_11 = var_10.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_11) == 1
    var_12 = var_10.__str__()
    assert var_12 == "{'d<ic*p#b=5n': 'd<ic*p#b=5n'}"
    var_13 = module_0.Message(text=var_11, position=var_0, end_position=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_13.text).__module__}.{type(var_13.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13.text) == 1
    assert var_13.code == 'custom'
    assert var_13.index == []
    assert var_13.start_position is None
    assert var_13.end_position is None
    var_14 = var_11.__eq__(var_11)
    assert var_14 is True
    var_15 = var_13.__repr__()
    assert var_15 == "Message(text=KeysView(BaseError([Message(text='d<ic*p#b=5n', code='custom', index=['d<ic*p#b=5n'])])), code='custom')"
    var_16 = var_13.__eq__(var_0)
    assert var_16 is False
    var_17 = var_10.__eq__(var_11)
    assert var_17 is False
    var_17.keys()

def test_case_44():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = False
    var_3 = module_0.Position(var_0, var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is False
    assert var_3.char_index is None
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = var_1.__eq__(var_0)
    assert var_5 is False
    var_6 = '&\rq9{@hw#FFN4'
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_6, messages=var_0)

def test_case_45():
    var_0 = 'Error'
    var_1 = 'test'
    var_2 = 'fie#d1'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'test'
    assert var_3.index == ['fie#d1']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 'field2'
    var_5 = module_0.Message(text=var_0, code=var_1, key=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Error'
    assert var_5.code == 'test'
    assert var_5.index == ['field2']
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_46():
    var_0 = 'vws;}&\x0b_p}'
    var_1 = None
    var_2 = None
    var_3 = None
    var_4 = module_0.ValidationError(text=var_0, code=var_0, key=var_0, position=var_3, messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__eq__(var_2)
    assert var_5 is False
    var_6 = var_5.__eq__(var_1)
    var_7 = var_4.__eq__(var_4)
    assert var_7 is True
    var_8 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is False
    var_7.__contains__(var_5)

def test_case_47():
    var_0 = '\r\\0c"-jNm='
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1, key=var_1, index=var_1, position=var_1, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == '\r\\0c"-jNm='
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_1, code=var_1, index=var_1, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False