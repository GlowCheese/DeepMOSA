# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is False
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    module_0.ParseError(code=var_0)

def test_case_1():
    var_0 = "k|)(&\x0c')0!k"
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == "k|)(&\x0c')0!k"
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None

def test_case_2():
    var_0 = '1;8LYW;VvvX$aRd_'
    var_1 = module_0.BaseError(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = repr(var_0)

def test_case_3():
    var_0 = 'Error'
    var_1 = 'required'
    var_2 = 'username'
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'required'
    assert var_3.index == ['username']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = repr(var_3)
    var_5 = "Message(text='Error', code='required', index=['username'])"
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_2, index=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'ValidationResult(value=None)'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text='ValidationResult(value=None)', code='custom')"
    module_0.ParseError(key=var_0)

def test_case_5():
    var_0 = 'Error'
    var_1 = 3
    var_2 = module_0.Message(text=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == 3
    assert var_2.end_position == 3
    var_3 = module_0.Message(text=var_0, position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == 3
    assert var_3.end_position == 3
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='Error', code='custom', position=3)"
    var_5 = var_2 == var_3
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_6():
    module_0.ParseError()

def test_case_7():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = 886
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 886
    assert var_1.column_no == 886
    assert var_1.char_index == 886
    module_0.ValidationError()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = "37'x3F'[8%GMD"
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.items()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_0, code=var_3, key=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == "37'x3F'[8%GMD"
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__repr__()
    assert var_5 == 'ItemsView(ValidationError(text="37\'x3F\'[8%GMD", code=\'custom\'))'
    var_6 = var_1.get(var_4)
    var_6.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    var_1 = '5~i0Fj3'
    var_2 = None
    var_3 = module_0.ParseError(text=var_1, code=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_3.__getitem__(var_0)

def test_case_11():
    var_0 = 'Error message'
    var_1 = 'error_code'
    var_2 = module_0.BaseError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = repr(var_2)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'
    var_3 = module_0.Message(text=var_1, index=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_3.text).__module__}.{type(var_3.text).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.code == 'custom'
    assert f'{type(var_3.index).__module__}.{type(var_3.index).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text=ValidationResult(value=None), code='custom', index=ValidationResult(value=None))"
    module_0.ParseError(messages=var_0)

def test_case_13():
    var_0 = 'Error'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_3.__repr__()
    assert var_4 == 'False'
    var_5 = 3
    var_6 = module_0.Message(text=var_0, position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position == 3
    assert var_6.end_position == 3
    var_7 = var_6.__hash__()
    assert var_7 == -5410329042497637944
    var_8 = module_0.Message(text=var_0, start_position=var_5, end_position=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == 3
    assert var_8.end_position == 3
    var_9 = var_6 == var_8
    assert var_9 is True

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = 'c!t}5"|zvMWW]z'
    var_2 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1.__eq__(var_0)
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = module_0.ValidationError(text=var_5, messages=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "Message(text=None, code='custom')"
    var_8 = var_6.__hash__()
    assert var_8 == 5149966352139022345
    var_9 = var_6.__eq__(var_0)
    assert var_9 is False
    var_10 = var_6.__contains__(var_0)
    assert var_10 is False
    var_10.__iter__()

def test_case_15():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Message(text=var_2, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ['key1', 'key2']
    assert var_3.code == 'key1'
    assert var_3.index == ['key1', 'key2']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__str__()
    assert var_6 == "{'key1': {'key2': ['key1', 'key2']}}"
    var_7 = repr(var_5)

def test_case_16():
    var_0 = 'Error'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error'
    assert var_2.code == 'code1'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'code2'
    var_4 = module_0.Message(text=var_0, code=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error'
    assert var_4.code == 'code2'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2 == var_4
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = []
    module_0.ParseError(messages=var_0)

def test_case_18():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Message(text=var_2, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ['key1', 'key2']
    assert var_3.code == 'key1'
    assert var_3.index == ['key1', 'key2']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = repr(var_5)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_1.__eq__(var_0)
    var_7 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.__str__()
    assert var_8 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_9 = var_7.__len__()
    assert var_9 == 1
    var_10 = var_7.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_10.__eq__(var_0)
    var_12 = None
    var_13 = True
    var_14 = module_0.Position(var_6, var_9, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_14.line_no).__module__}.{type(var_14.line_no).__qualname__}' == 'builtins.NotImplementedType'
    assert var_14.column_no == 1
    assert var_14.char_index is True
    var_15 = var_10.__eq__(var_0)
    var_16 = module_0.ValidationResult(value=var_10)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_16.value).__module__}.{type(var_16.value).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16.value) == 1
    assert var_16.error is None
    var_17 = var_10.__repr__()
    assert var_17 == 'KeysView(ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\'))'
    var_18 = module_0.Message(text=var_12)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text is None
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert var_18.start_position is None
    assert var_18.end_position is None
    var_19 = var_4.__eq__(var_12)
    assert var_19 is False
    var_20 = var_18.__repr__()
    assert var_20 == "Message(text=None, code='custom')"
    module_0.ParseError(text=var_0)

def test_case_20():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_6 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_5.__eq__(var_0)
    var_8 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = var_8.__hash__()
    assert var_9 == 5149966352139022345
    var_10 = var_8.__eq__(var_0)
    assert var_10 is False
    var_11 = var_8.__contains__(var_10)
    assert var_11 is False
    var_12 = var_8.__iter__()
    var_13 = var_8.keys()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_13) == 1
    var_14 = var_8.__repr__()
    assert var_14 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_15 = var_8.__iter__()
    var_16 = var_8.__iter__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, code=var_14, key=var_11, index=var_13, start_position=var_11)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 1
    module_0.ParseError(key=var_0, messages=var_0)

def test_case_22():
    var_0 = 'Error'
    var_1 = 'custom'
    var_2 = []
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = repr(var_3)
    var_5 = "Message(text='Error', code='custom')"
    var_6 = bool(var_4 == var_5)
    assert var_6 is True

def test_case_23():
    var_0 = None
    var_1 = '7\x0cV+\x0btAL1ab#.\\'
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.messages(add_prefix=var_0)

def test_case_24():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 5
    assert var_3.char_index == 10
    var_4 = 'invalid'
    var_5 = module_0.Message(text=var_4, code=var_4, position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'invalid'
    assert var_5.code == 'invalid'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = repr(var_5)

def test_case_25():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_6 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text is None
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_5.__eq__(var_0)
    var_8 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_10 = var_8.__hash__()
    assert var_10 == 5149966352139022345
    var_11 = var_8.__eq__(var_0)
    assert var_11 is False
    var_12 = var_6.__eq__(var_0)
    assert var_12 is False
    var_13 = var_8.__contains__(var_11)
    assert var_13 is False
    var_14 = var_8.__iter__()
    var_15 = var_8.keys()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_15) == 1
    var_16 = var_8.__repr__()
    assert var_16 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_17 = var_8.__iter__()
    var_18 = var_8.__iter__()
    var_19 = module_0.Message(text=var_11, position=var_13, start_position=var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is False
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is False
    assert var_19.end_position is False
    var_20 = var_8.__repr__()
    assert var_20 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_21 = var_8.__repr__()
    assert var_21 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_22 = module_0.Position(var_0, var_13, var_11)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert var_22.line_no is None
    assert var_22.column_no is False
    assert var_22.char_index is False
    var_23 = ''
    with pytest.raises(AssertionError):
        module_0.Message(text=var_23, code=var_11, key=var_0, position=var_15, end_position=var_11)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = True
    var_8 = module_0.Position(var_0, var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is True
    assert var_8.char_index is True
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_5.__eq__(var_0)
    var_12 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_12.messages()
    var_14 = var_12.values()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_14) == 1
    var_15 = var_12.__str__()
    assert var_15 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_16 = var_12.__hash__()
    assert var_16 == 5149966352139022345
    var_17 = var_2.__hash__()
    assert var_17 == -5410329042497637944
    var_18 = var_12.__eq__(var_0)
    assert var_18 is False
    var_19 = var_8.__eq__(var_18)
    assert var_19 is False
    var_20 = module_0.ValidationResult(value=var_1)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_20.value == 'c!t}5"|=vMWW]z'
    assert var_20.error is None
    var_21 = var_20.__repr__()
    assert var_21 == 'ValidationResult(value=\'c!t}5"|=vMWW]z\')'
    var_22 = None
    var_23 = 'L3`T|\n&W\\'
    var_24 = module_0.Message(text=var_23, index=var_22, end_position=var_18)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Message'
    assert var_24.text == 'L3`T|\n&W\\'
    assert var_24.code == 'custom'
    assert var_24.index == []
    assert var_24.start_position is None
    assert var_24.end_position is False
    module_0.ParseError(text=var_18, code=var_15, key=var_18, position=var_22, messages=var_13)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = var_2.__repr__()
    assert var_4 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_5 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_1)
    assert var_6 is False
    var_7 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8) == 1
    var_9 = var_7.__len__()
    assert var_9 == 1
    var_10 = var_7.__eq__(var_0)
    assert var_10 is False
    var_11 = var_5.__eq__(var_0)
    assert var_11 is False
    var_12 = var_7.__contains__(var_10)
    assert var_12 is False
    var_13 = var_7.__contains__(var_10)
    assert var_13 is False
    var_14 = var_7.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    module_0.ParseError(code=var_9, messages=var_10)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = var_2.__repr__()
    assert var_4 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_5 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_3.__repr__()
    assert var_6 == '\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\''
    var_7 = var_4.__eq__(var_0)
    var_8 = var_2.__hash__()
    assert var_8 == -5410329042497637944
    var_9 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_9.__str__()
    assert var_10 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_11 = var_9.__hash__()
    assert var_11 == 5149966352139022345
    var_12 = var_9.__eq__(var_0)
    assert var_12 is False
    var_13 = var_5.__eq__(var_2)
    assert var_13 is False
    var_14 = var_9.__contains__(var_0)
    assert var_14 is False
    var_14.__iter__()

def test_case_29():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
    assert var_1.error is None
    var_2 = iter(var_1)
    var_3 = next(var_2)
    var_4 = next(var_2)
    assert var_4 is None
    var_5 = bool(var_3 == var_0)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_1.__eq__(var_0)
    var_7 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_7.__str__()
    assert var_8 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_9 = var_7.__len__()
    assert var_9 == 1
    var_10 = var_7.__eq__(var_0)
    assert var_10 is False
    var_11 = var_5.__eq__(var_0)
    assert var_11 is False
    var_12 = var_7.__hash__()
    assert var_12 == 5149966352139022345
    var_13 = var_7.__contains__(var_10)
    assert var_13 is False
    var_14 = var_7.__iter__()
    var_15 = var_7.keys()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_15) == 1
    var_16 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_7.__repr__()
    assert var_17 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_18 = var_7.__iter__()
    var_19 = var_7.__iter__()
    var_20 = module_0.Message(text=var_10, position=var_13, start_position=var_0)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text is False
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert var_20.start_position is False
    assert var_20.end_position is False
    var_21 = var_7.__repr__()
    assert var_21 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_22 = var_7.__str__()
    assert var_22 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_23 = var_7.__repr__()
    assert var_23 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_24 = var_20.__hash__()
    assert var_24 == -5410329042497637944
    var_25 = module_0.Message(text=var_13, index=var_10)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.Message'
    assert var_25.text is False
    assert var_25.code == 'custom'
    assert var_25.index is False
    assert var_25.start_position is None
    assert var_25.end_position is None
    module_0.ParseError(position=var_10, messages=var_13)

def test_case_31():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_1.__eq__(var_0)
    var_6 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_8 = var_6.__len__()
    assert var_8 == 1
    var_9 = var_6.__eq__(var_0)
    assert var_9 is False
    var_10 = var_4.__eq__(var_0)
    assert var_10 is False
    var_11 = var_6.__hash__()
    assert var_11 == 5149966352139022345
    var_12 = var_6.__contains__(var_9)
    assert var_12 is False
    var_13 = var_6.__iter__()
    var_14 = var_6.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is None
    var_16 = var_6.__repr__()
    assert var_16 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_17 = var_6.__iter__()
    var_18 = var_6.__iter__()
    var_19 = module_0.Message(text=var_9, position=var_12, start_position=var_0)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text is False
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is False
    assert var_19.end_position is False
    var_20 = var_6.__repr__()
    assert var_20 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_21 = var_6.__str__()
    assert var_21 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_22 = var_6.__iter__()
    var_23 = var_14.__iter__()
    var_24 = module_0.Message(text=var_0, code=var_1, start_position=var_9)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.Message'
    assert var_24.text is None
    assert var_24.code == 'c!t}5"|=vMWW]z'
    assert var_24.index == []
    assert var_24.start_position is False
    assert var_24.end_position is None
    var_25 = var_6.__repr__()
    assert var_25 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_26 = var_6.__repr__()
    assert var_26 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_27 = var_19.__eq__(var_12)
    assert var_27 is False
    var_28 = 'DOL8Msjc\\,_)'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_28, position=var_9, start_position=var_14)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = True
    var_8 = module_0.Position(var_0, var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is True
    assert var_8.char_index is True
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_5.__eq__(var_0)
    var_12 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_12.messages()
    var_14 = var_12.__eq__(var_12)
    assert var_14 is True
    var_15 = var_12.__str__()
    assert var_15 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_16 = var_12.__hash__()
    assert var_16 == 5149966352139022345
    var_17 = var_2.__hash__()
    assert var_17 == -5410329042497637944
    var_18 = var_12.__eq__(var_0)
    assert var_18 is False
    var_19 = var_10.__repr__()
    assert var_19 == "Message(text=None, code='custom')"
    var_20 = var_12.__hash__()
    assert var_20 == 5149966352139022345
    var_21 = var_10.__eq__(var_0)
    assert var_21 is False
    var_22 = var_12.__contains__(var_18)
    assert var_22 is False
    var_23 = var_12.__iter__()
    var_24 = var_12.keys()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_24) == 1
    var_25 = var_12.__repr__()
    assert var_25 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_26 = var_12.__iter__()
    var_27 = var_12.__iter__()
    var_28 = module_0.Message(text=var_18, position=var_22, start_position=var_0)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Message'
    assert var_28.text is False
    assert var_28.code == 'custom'
    assert var_28.index == []
    assert var_28.start_position is False
    assert var_28.end_position is False
    var_29 = var_12.__repr__()
    assert var_29 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_30 = var_12.__repr__()
    assert var_30 == 'ValidationError(text=\'Message(text=\\\'c!t}5"|=vMWW]z\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_31 = var_18.__eq__(var_18)
    assert var_31 is True
    var_32 = 344
    var_33 = '7_2yy!1Uv'
    var_34 = module_0.Message(text=var_33, code=var_0, key=var_32)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.base.Message'
    assert var_34.text == '7_2yy!1Uv'
    assert var_34.code == 'custom'
    assert var_34.index == [344]
    assert var_34.start_position is None
    assert var_34.end_position is None
    module_0.ParseError(key=var_24, position=var_0, messages=var_24)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_4 = module_0.Message(text=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = False
    var_8 = True
    var_9 = module_0.Position(var_0, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no is False
    assert var_9.char_index is True
    var_10 = var_9.__eq__(var_0)
    assert var_10 is False
    var_11 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is None
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_9.__eq__(var_0)
    assert var_12 is False
    var_13 = module_0.ValidationError(text=var_6, code=var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13) == 1
    var_14 = var_13.messages(add_prefix=var_0)
    var_15 = var_13.messages(add_prefix=var_7)
    var_16 = var_13.__str__()
    assert var_16 == "Message(text=None, code='custom')"
    var_17 = var_13.items()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17) == 1
    var_18 = var_9.__eq__(var_17)
    assert var_18 is False
    var_19 = var_13.__hash__()
    assert var_19 == 5149966352139022345
    var_20 = var_4.__hash__()
    assert var_20 == -5410329042497637944
    var_21 = var_13.__eq__(var_0)
    assert var_21 is False
    var_22 = var_13.items()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_22) == 1
    var_22.__hash__()

def test_case_34():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Message(text=var_2, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ['key1', 'key2']
    assert var_3.code == 'key1'
    assert var_3.index == ['key1', 'key2']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3]
    var_5 = module_0.BaseError(messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__iter__()
    var_7 = var_5.keys()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7) == 1
    var_8 = var_7.__repr__()
    assert var_8 == "KeysView(BaseError([Message(text=['key1', 'key2'], code='key1', index=['key1', 'key2'])]))"
    var_9 = var_5.__str__()
    assert var_9 == "{'key1': {'key2': ['key1', 'key2']}}"
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_7, error=var_7)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = 'c!t}5\\|=vMWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5\\|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='c!t}5\\\\|=vMWW]z', code='custom')"
    var_4 = module_0.Message(text=var_0, key=var_0, start_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text='c!t}5\\\\|=vMWW]z', code='custom')"
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text=None, code='custom')"
    var_7 = True
    var_8 = module_0.Position(var_0, var_7, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is True
    assert var_8.char_index is True
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_5.__eq__(var_0)
    var_12 = module_0.ValidationError(text=var_3, messages=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_12.messages()
    var_14 = var_12.messages()
    var_15 = var_12.__str__()
    assert var_15 == "Message(text='c!t}5\\\\|=vMWW]z', code='custom')"
    var_16 = var_12.__hash__()
    assert var_16 == 5149966352139022345
    var_17 = var_2.__hash__()
    assert var_17 == -5410329042497637944
    var_18 = var_12.__eq__(var_0)
    assert var_18 is False
    var_19 = var_8.__eq__(var_18)
    assert var_19 is False
    var_20 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_20.value is None
    assert f'{type(var_20.error).__module__}.{type(var_20.error).__qualname__}' == 'typesystem.base.Message'
    var_21 = var_20.__repr__()
    assert var_21 == "ValidationResult(error=Message(text='c!t}5\\\\|=vMWW]z', code='custom'))"
    var_22 = None
    var_23 = module_0.Message(text=var_11, code=var_5)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_23.text).__module__}.{type(var_23.text).__qualname__}' == 'builtins.NotImplementedType'
    assert var_23.code == "Message(text='c!t}5\\\\|=vMWW]z', code='custom')"
    assert var_23.index == []
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = var_12.__hash__()
    assert var_24 == 5149966352139022345
    var_25 = var_2.__eq__(var_18)
    assert var_25 is False
    var_26 = var_12.__contains__(var_22)
    assert var_26 is False
    var_27 = var_26.__repr__()
    assert var_27 == 'False'
    module_0.ParseError()

def test_case_36():
    var_0 = 'Error'
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = module_0.Message(text=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'custom'
    assert var_3.index == ['Error', 1]
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = 2
    var_5 = [var_0, var_4]
    var_6 = module_0.Message(text=var_0, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error'
    assert var_6.code == 'custom'
    assert var_6.index == ['Error', 2]
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_3 == var_6
    assert var_7 is False

def test_case_37():
    var_0 = 'Error'
    var_1 = 1
    var_2 = 5
    var_3 = module_0.Message(text=var_0, start_position=var_1, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == 1
    assert var_3.end_position == 5
    var_4 = 2
    var_5 = module_0.Message(text=var_0, start_position=var_4, end_position=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Error'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position == 2
    assert var_5.end_position == 5
    var_6 = var_3 == var_5
    assert var_6 is False

def test_case_38():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = 4
    var_5 = module_0.Position(var_4, var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 4
    assert var_5.column_no == 2
    assert var_5.char_index == 3
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_39():
    var_0 = 1
    var_1 = 5
    var_2 = 8
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 5
    assert var_3.char_index == 8
    var_4 = 15
    var_5 = module_0.Position(var_0, var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 8
    assert var_5.char_index == 15
    var_6 = 'Error'
    var_7 = 'invalid'
    var_8 = module_0.Message(text=var_6, code=var_7, start_position=var_3, end_position=var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'Error'
    assert var_8.code == 'invalid'
    assert var_8.index == []
    assert f'{type(var_8.start_position).__module__}.{type(var_8.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_8.end_position).__module__}.{type(var_8.end_position).__qualname__}' == 'typesystem.base.Position'
    var_9 = repr(var_8)
    var_10 = "Message(text='Error', code='invalid', start_position=Position(line_no=1, column_no=5, char_index=10), end_position=Position(line_no=1, column_no=10, char_index=15))"
    var_11 = bool(var_9 == var_10)

def test_case_40():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = 6
    var_5 = module_0.Position(var_0, var_1, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 2
    assert var_5.char_index == 6
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_41():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 2
    assert var_3.char_index == 3
    var_4 = 5
    var_5 = module_0.Position(var_0, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 5
    assert var_5.char_index == 3
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_42():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error 1'
    assert var_2.code == 'code1'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'Error 2'
    var_4 = 'code2'
    var_5 = 'key'
    var_6 = [var_5]
    var_7 = module_0.Message(text=var_3, code=var_4, index=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Error 2'
    assert var_7.code == 'code2'
    assert var_7.index == ['key']
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = [var_2, var_7]
    var_9 = module_0.BaseError(messages=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 2
    var_10 = repr(var_9)
    var_11 = bool(var_10 == var_10)
    assert var_11 is True

def test_case_43():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = [var_0, var_1]
    var_3 = module_0.Message(text=var_2, code=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == ['key1', 'key2']
    assert var_3.code == 'key1'
    assert var_3.index == ['key1', 'key2']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text=['key1', 'key2'], code='key1', index=['key1', 'key2'])"
    var_5 = [var_3, var_3, var_3]
    var_6 = module_0.BaseError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == "{'key1': {'key2': ['key1', 'key2']}}"
    var_8 = repr(var_3)
    var_9 = module_0.ValidationResult(value=var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == "{'key1': {'key2': ['key1', 'key2']}}"
    assert var_9.error is None