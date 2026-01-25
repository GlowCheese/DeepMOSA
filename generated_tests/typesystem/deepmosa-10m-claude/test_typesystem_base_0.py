# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is None
    assert var_2.char_index is False
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = var_2.__eq__(var_0)
    assert var_4 is False

def test_case_1():
    var_0 = '(FE^C'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

def test_case_2():
    var_0 = 'bEA6lj\x0by(~7Rwm1D_*)'
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'bEA6lj\x0by(~7Rwm1D_*)'
    assert var_1.code == 'bEA6lj\x0by(~7Rwm1D_*)'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_3():
    var_0 = "C1{'g"
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1

def test_case_4():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_2.__hash__()
    assert var_4 == 9180260228493912499
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = []
    module_0.ValidationError(messages=var_0)

def test_case_6():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0, error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__bool__()
    assert var_2 is True

def test_case_7():
    var_0 = 1
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 1
    assert var_1.column_no == 1
    assert var_1.char_index == 1

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = False
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__hash__()
    assert var_3 == 8224402618462540757
    var_3.get(var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = False
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()
    var_3.get(var_1)

def test_case_10():
    var_0 = '(FE^C'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_3)
    assert var_4 is True

def test_case_11():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = "C1{'g"
    var_3 = module_0.ParseError(text=var_2, key=var_2, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__str__()
    assert var_4 == '{"C1{\'g": "C1{\'g"}'
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    module_0.ValidationError(key=var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'B(`!o+n(IF~Xz/)k8'
    var_1 = None
    var_2 = module_0.Message(text=var_0, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'B(`!o+n(IF~Xz/)k8'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = [var_2, var_2, var_2]
    module_0.ParseError(code=var_0, messages=var_3)

def test_case_14():
    var_0 = True
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is True
    assert var_1.code == 'custom'
    assert var_1.index == [True]
    assert var_1.start_position is True
    assert var_1.end_position is True
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=True, code='custom', index=[True], position=True)"
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=True, code='custom', index=[True], position=True)"

def test_case_15():
    var_0 = 'FEh^C'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.values()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "ValuesView(ValidationError(text='FEh^C', code='custom'))"
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True

def test_case_16():
    var_0 = "C1{'g"
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1
    var_4 = var_3.__repr__()
    assert var_4 == 'ItemsView(ParseError([Message(text="C1{\'g", code=\'custom\', index=["C1{\'g"])]))'

def test_case_17():
    var_0 = '(FE^C'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()
    var_3 = var_1.items()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_3) == 1

def test_case_18():
    var_0 = False
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is False
    assert var_1.column_no is False
    assert var_1.char_index is False
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

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

def test_case_20():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = [var_1]
    var_3 = module_0.ParseError(messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = var_3.__eq__(var_0)
    assert var_5 is False
    var_6 = var_1.__eq__(var_0)
    assert var_6 is False

def test_case_21():
    var_0 = 'x'
    var_1 = module_0.Message(text=var_0, code=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == 'x'
    assert var_1.code == 'x'
    assert var_1.index == []
    assert var_1.start_position == 'x'
    assert var_1.end_position == 'x'

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 9180260228493912499
    var_4 = "C1{'g"
    module_0.ParseError(text=var_4, key=var_1, messages=var_2)

def test_case_23():
    var_0 = '(FE^C'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1
    var_3 = False
    var_4 = module_0.ValidationError(text=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4) == 1
    var_5 = var_4.__contains__(var_4)
    assert var_5 is False
    var_6 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is False
    assert var_6.error is None

def test_case_24():
    var_0 = 'Error'
    var_1 = 'error_code'
    var_2 = [var_0]
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Error'
    assert var_3.code == 'error_code'
    assert var_3.index == ['Error']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = bool(var_3 != var_3)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = '(FE^C'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_2.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "ItemsView(ValidationError(text='(FE^C', code='custom'))"
    module_0.ParseError(key=var_4, position=var_1, messages=var_4)

def test_case_26():
    var_0 = '(FE^C'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_2.items()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "ItemsView(ValidationError(text='(FE^C', code='custom'))"
    var_6 = var_2.__str__()
    assert var_6 == '(FE^C'
    var_7 = module_0.Message(text=var_5, code=var_4, index=var_1, end_position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == "ItemsView(ValidationError(text='(FE^C', code='custom'))"
    assert f'{type(var_7.code).__module__}.{type(var_7.code).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_7.code) == 1
    assert var_7.index == []
    assert var_7.start_position is None
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_7.end_position) == 1
    var_8 = var_7.__eq__(var_4)
    assert var_8 is False

def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = None
    var_4 = '\n-6j@b?[!UA8'
    var_5 = module_0.Message(text=var_4, code=var_3, position=var_4, start_position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '\n-6j@b?[!UA8'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position == '\n-6j@b?[!UA8'
    assert var_5.end_position == '\n-6j@b?[!UA8'
    var_6 = var_5.__eq__(var_3)
    assert var_6 is False
    var_7 = module_0.Message(text=var_4, end_position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == '\n-6j@b?[!UA8'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__repr__()
    assert var_8 == "Message(text='\\n-6j@b?[!UA8', code='custom')"
    var_9 = var_1.__eq__(var_5)
    assert var_9 is False

def test_case_28():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == 9180260228493912499
    var_3 = "C1{'g"
    var_4 = module_0.ParseError(text=var_3, key=var_0, messages=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_4) == 1
    var_5 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == 9180260228493912499
    assert var_5.error is None
    var_6 = 'x'
    var_7 = var_4.messages(add_prefix=var_6)
    var_8 = var_4.__str__()
    assert var_8 == "C1{'g"
    var_9 = var_4.__eq__(var_0)
    assert var_9 is False

def test_case_29():
    var_0 = 1648
    var_1 = None
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 1648
    assert var_2.column_no == 1648
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False
    var_4 = var_2.__repr__()
    assert var_4 == 'Position(line_no=1648, column_no=1648, char_index=None)'
    var_5 = 'Z94qyl@A+.R<UXK^'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, code=var_4, index=var_1, position=var_2, start_position=var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = 1169
    var_2 = False
    var_3 = module_0.Position(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1169
    assert var_3.column_no is None
    assert var_3.char_index is False
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__hash__()
    assert var_5 == 9180260228493912499
    var_6 = "C1{'g"
    var_7 = module_0.BaseError(text=var_6, code=var_6, key=var_0, messages=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = module_0.Message(text=var_0, code=var_6, key=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == "C1{'g"
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_4.__eq__(var_8)
    assert var_9 is False
    var_10 = module_0.Message(text=var_0, code=var_0, position=var_0, end_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    module_0.ValidationError(text=var_0, position=var_0, messages=var_0)

def test_case_31():
    var_0 = None
    var_1 = 1141
    var_2 = False
    var_3 = module_0.Position(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1141
    assert var_3.column_no is None
    assert var_3.char_index is False
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = "C1{'g"
    var_7 = module_0.BaseError(text=var_6, code=var_6, key=var_0, messages=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_7) == 1
    var_8 = var_7.keys()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_8) == 1
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_4, error=var_8)

def test_case_32():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = '\n-6j@b?[!UA8'
    var_7 = 'gut{j+V0y,?'
    var_8 = module_0.Message(text=var_7, code=var_5, position=var_6, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'gut{j+V0y,?'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == '\n-6j@b?[!UA8'
    assert var_8.end_position == '\n-6j@b?[!UA8'
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.Message(text=var_6, end_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == '\n-6j@b?[!UA8'
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_2.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = [var_10, var_2, var_8]
    var_13 = module_0.ParseError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.__eq__(var_5)
    assert var_14 is False
    var_15 = var_13.values()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_15) == 1
    var_16 = module_0.Message(text=var_3)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = var_13.__str__()
    assert var_17 == "{'': 'gut{j+V0y,?'}"

def test_case_33():
    var_0 = None
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    var_4 = var_2.__eq__(var_3)
    assert var_4 is False
    var_5 = None
    var_6 = '\n-6j@b?[!UA8'
    var_7 = 'gut{j+V0y,?'
    var_8 = module_0.Message(text=var_7, code=var_5, position=var_6, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'gut{j+V0y,?'
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == '\n-6j@b?[!UA8'
    assert var_8.end_position == '\n-6j@b?[!UA8'
    var_9 = var_8.__eq__(var_0)
    assert var_9 is False
    var_10 = module_0.Message(text=var_6, end_position=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == '\n-6j@b?[!UA8'
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_2.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_12 = [var_10, var_2, var_8]
    var_13 = module_0.ParseError(messages=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.__eq__(var_5)
    assert var_14 is False
    var_15 = var_13.values()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_15) == 1
    var_16 = module_0.Message(text=var_3)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = var_15.__str__()
    assert var_17 == "ValuesView(ParseError([Message(text='\\n-6j@b?[!UA8', code='custom'), Message(text=None, code='custom'), Message(text='gut{j+V0y,?', code='custom', position='\\n-6j@b?[!UA8')]))"

def test_case_34():
    var_0 = None
    var_1 = False
    var_2 = module_0.Position(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is False
    assert var_2.column_no is None
    assert var_2.char_index is False
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__eq__(var_4)
    assert var_5 is True
    var_6 = "C1{'g"
    var_7 = [var_4]
    var_8 = module_0.BaseError(text=var_6, code=var_6, key=var_0, messages=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8.keys()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_9) == 1
    var_10 = var_9.__repr__()
    assert var_10 == 'KeysView(BaseError(text="C1{\'g", code="C1{\'g"))'
    var_11 = var_9.__eq__(var_9)
    assert var_11 is True
    var_12 = var_8.values()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_12) == 1
    var_13 = 'Ez8b!] '
    with pytest.raises(AssertionError):
        module_0.Message(text=var_13, position=var_7, end_position=var_12)

def test_case_35():
    var_0 = None
    var_1 = 1169
    var_2 = True
    var_3 = module_0.Position(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1169
    assert var_3.column_no is None
    assert var_3.char_index is True
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = "C1{'g"
    var_8 = var_3.__repr__()
    assert var_8 == 'Position(line_no=1169, column_no=None, char_index=True)'
    var_9 = module_0.BaseError(text=var_7, code=var_7, key=var_0, messages=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == 'KeysView(BaseError(text="C1{\'g", code="C1{\'g"))'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_10, key=var_4, index=var_10, start_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = 1169
    var_2 = True
    var_3 = module_0.Position(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1169
    assert var_3.column_no is None
    assert var_3.char_index is True
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = "C1{'g"
    var_8 = [var_5]
    var_9 = module_0.BaseError(text=var_7, code=var_7, key=var_0, messages=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_9) == 1
    var_10 = var_9.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == 'KeysView(BaseError(text="C1{\'g", code="C1{\'g"))'
    var_12 = module_0.ParseError(messages=var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_10.__repr__()
    assert var_13 == 'KeysView(BaseError(text="C1{\'g", code="C1{\'g"))'
    var_14 = var_5.__repr__()
    assert var_14 == "Message(text=None, code='custom')"
    module_0.ValidationError(position=var_10, messages=var_10)

def test_case_37():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

def test_case_38():
    var_0 = None
    var_1 = 895
    var_2 = False
    var_3 = module_0.Position(var_1, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 895
    assert var_3.column_no is None
    assert var_3.char_index is False
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = "C1{'g"
    var_8 = [var_5]
    var_9 = module_0.ValidationError(text=var_7, code=var_7, key=var_2, position=var_3, messages=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9) == 1
    var_10 = var_9.keys()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_10) == 1
    var_11 = var_10.__repr__()
    assert var_11 == 'KeysView(ValidationError([Message(text="C1{\'g", code="C1{\'g", index=[False], position=Position(line_no=895, column_no=None, char_index=False))]))'
    var_12 = module_0.ParseError(code=var_0, messages=var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_10.__eq__(var_0)
    var_14 = var_9.__eq__(var_9)
    assert var_14 is True
    var_15 = var_10.__str__()
    assert var_15 == 'KeysView(ValidationError([Message(text="C1{\'g", code="C1{\'g", index=[False], position=Position(line_no=895, column_no=None, char_index=False))]))'

def test_case_39():
    var_0 = 'Err'
    var_1 = 'error_code'
    var_2 = [var_0]
    var_3 = module_0.Message(text=var_0, code=var_1, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'Err'
    assert var_3.code == 'error_code'
    assert var_3.index == ['Err']
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = []
    var_5 = module_0.Message(text=var_0, code=var_1, index=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == 'Err'
    assert var_5.code == 'error_code'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = bool(var_3 != var_5)
    assert var_6 is True

def test_case_40():
    var_0 = 5
    var_1 = 10
    var_2 = module_0.Position(var_1, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 10
    assert var_2.column_no == 5
    assert var_2.char_index == 10
    var_3 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 5
    assert var_3.column_no == 5
    assert var_3.char_index == 5
    var_4 = bool(not var_2 == var_3)
    assert var_4 is True

def test_case_41():
    var_0 = 1
    var_1 = 5
    var_2 = 10
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1
    assert var_3.column_no == 5
    assert var_3.char_index == 10
    var_4 = 6
    var_5 = module_0.Position(var_0, var_4, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 1
    assert var_5.column_no == 6
    assert var_5.char_index == 10
    var_6 = bool(not var_3 == var_5)
    assert var_6 is True

def test_case_42():
    var_0 = 'test_data'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_data'
    assert var_1.error is None
    var_2 = 'Test error'
    var_3 = module_0.ValidationError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = module_0.ValidationResult(error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = 42
    var_6 = module_0.ValidationResult(value=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value == 42
    assert var_6.error is None
    var_7 = iter(var_6)
    var_8 = next(var_7)
    assert var_8 == 42
    var_9 = next(var_7)
    assert var_9 is None

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 901
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error == 901
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(error=901)'
    module_0.ParseError()

def test_case_44():
    var_0 = 'Invalid email'
    var_1 = 'invalid_email'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'email'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Invalid email'
    assert var_6.code == 'invalid_email'
    assert var_6.index == ['users', 0, 'email']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = [var_6]
    var_8 = module_0.BaseError(messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 1
    var_9 = var_8._message_dict['users'][0]['email']
    assert var_9 == 'Invalid email'

@pytest.mark.xfail(strict=True)
def test_case_45():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Message(text=var_0, start_position=var_2, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is False
    assert var_3.end_position is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text=None, code='custom', start_position=False, end_position=None)"
    module_0.ValidationError()

def test_case_46():
    var_0 = None
    var_1 = None
    var_2 = '\n-6j@b?[!UA8'
    var_3 = module_0.Message(text=var_2, code=var_1, position=var_2, start_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '\n-6j@b?[!UA8'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == '\n-6j@b?[!UA8'
    assert var_3.end_position == '\n-6j@b?[!UA8'
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = module_0.Message(text=var_2, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '\n-6j@b?[!UA8'
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text='\\n-6j@b?[!UA8', code='custom')"
    var_7 = var_5.__eq__(var_3)
    assert var_7 is False