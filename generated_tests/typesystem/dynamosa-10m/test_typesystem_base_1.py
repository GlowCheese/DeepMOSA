# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_1():
    var_0 = 'zKpZGI[/o)7'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1

def test_case_2():
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

@pytest.mark.xfail(strict=True)
def test_case_3():
    module_0.ValidationError()

def test_case_4():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_5():
    var_0 = 'l\r3,w,\r1'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    assert var_2 == -1502684650933579160

def test_case_6():
    var_0 = 'l\r3,w,\r1'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__contains__(var_1)
    assert var_2 is False

def test_case_7():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True

def test_case_8():
    var_0 = 'dE)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

def test_case_9():
    var_0 = 'dE)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages(add_prefix=var_0)

def test_case_10():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = 'J\nJ>NaOip%:MIEG?'
    module_0.BaseError(messages=var_0)

def test_case_12():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None

def test_case_13():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=None, column_no=None, char_index=None)'

def test_case_14():
    var_0 = 'GB)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

def test_case_15():
    var_0 = 'e'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "ValidationError(text='e', code='custom')"

def test_case_16():
    var_0 = -69
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == -69

def test_case_17():
    var_0 = 'dE)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1

def test_case_18():
    var_0 = 'EdqSon!6u\\U^..8-gB'
    var_1 = module_0.ParseError(text=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1

def test_case_19():
    var_0 = '3[|5\x0cGi"C\\DlYJR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_0, position=var_1, start_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '3[|5\x0cGi"C\\DlYJR'
    assert var_2.code == 'custom'
    assert var_2.index == ['3[|5\x0cGi"C\\DlYJR']
    assert var_2.start_position == '3[|5\x0cGi"C\\DlYJR'
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'3[|5\\x0cGi"C\\\\DlYJR\', code=\'custom\', index=[\'3[|5\\x0cGi"C\\\\DlYJR\'], start_position=\'3[|5\\x0cGi"C\\\\DlYJR\', end_position=None)'

def test_case_20():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__repr__()
    assert var_1 == 'ValidationResult(value=None)'

def test_case_21():
    var_0 = 'GB)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = '097Bf,!znU^<D6'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__len__()
    assert var_3 == 1
    var_4 = module_0.Message(text=var_1, key=var_3, index=var_1, position=var_3, start_position=var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == [1]
    assert var_4.start_position == 1
    assert var_4.end_position == 1
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text=None, code='custom', index=[1], position=1)"
    var_6 = var_2.__repr__()
    assert var_6 == "ValidationError(text='097Bf,!znU^<D6', code='custom')"
    var_7 = var_3.__eq__(var_3)
    assert var_7 is True
    var_8 = var_2.__contains__(var_5)
    assert var_8 is False
    var_8.__iter__()

def test_case_23():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

def test_case_24():
    var_0 = 'c!t}5"|=vMWW]z'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = True
    module_0.ValidationError(code=var_0, messages=var_0)

def test_case_26():
    var_0 = 'c!t}5|=vMWM]z'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'c!t}5|=vMWM]z'
    assert var_1.error is None

def test_case_27():
    var_0 = 'LH\\6Y&ii0:wzz?\t'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == "{'LH\\\\6Y&ii0:wzz?\\t': 'LH\\\\6Y&ii0:wzz?\\t'}"

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = False
    module_0.ParseError(key=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'xz^ '
    var_1 = []
    module_0.ParseError(text=var_0, code=var_0, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = -1106.7005
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    module_0.ValidationError(position=var_0, messages=var_0)

def test_case_31():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_32():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.Message(text=var_2, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "Message(text=None, code='custom')"
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.__repr__()
    assert var_4 == '"Message(text=None, code=\'custom\')"'
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False

def test_case_33():
    var_0 = 'LH\\6Y&ii0:wzz?\t'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "BaseError([Message(text='LH\\\\6Y&ii0:wzz?\\t', code='custom', index=['LH\\\\6Y&ii0:wzz?\\t'])])"
    var_3 = var_1.__len__()
    assert var_3 == 1

def test_case_34():
    var_0 = None
    var_1 = 'r.F ?{imQNQ5'
    var_2 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index is None
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_2, error=var_1)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = None
    var_1 = []
    module_0.ValidationError(key=var_0, position=var_0, messages=var_1)

def test_case_36():
    var_0 = '3[5\x0cGi"C\\DlYJR'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, position=var_0, start_position=var_0)

def test_case_37():
    var_0 = None
    var_1 = -1190
    var_2 = module_0.Position(var_0, var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no == -1190
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

def test_case_38():
    var_0 = None
    var_1 = 'c!t}5"|=vWW]z'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    with pytest.raises(AssertionError):
        module_0.Message(text=var_1, index=var_2, position=var_2, end_position=var_3)

def test_case_39():
    var_0 = 'GB)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()

def test_case_40():
    var_0 = 'r.F ?{imQNQ5'
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error == 'r.F ?{imQNQ5'
    var_2 = var_1.__repr__()
    assert var_2 == "ValidationResult(error='r.F ?{imQNQ5')"

def test_case_41():
    var_0 = None
    var_1 = 'r.F ?{imQNQ5'
    var_2 = module_0.Message(text=var_1, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'r.F ?{imQNQ5'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 7078395332879544620
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = [var_2, var_2, var_2]
    var_6 = module_0.ParseError(code=var_0, position=var_0, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_6.values()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_7) == 1
    var_8 = module_0.Message(text=var_1, code=var_7, position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text == 'r.F ?{imQNQ5'
    assert f'{type(var_8.code).__module__}.{type(var_8.code).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8.code) == 1
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_2.__eq__(var_8)
    assert var_9 is False
    var_10 = var_8.__hash__()

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = None
    var_1 = 'r.F ?{imQNQ5'
    var_2 = module_0.Message(text=var_1, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'r.F ?{imQNQ5'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False
    var_4 = var_2.__hash__()
    assert var_4 == 7078395332879544620
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = [var_2, var_2, var_2]
    var_7 = module_0.ParseError(code=var_0, position=var_0, messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_7.__repr__()
    assert var_8 == "ParseError([Message(text='r.F ?{imQNQ5', code='custom'), Message(text='r.F ?{imQNQ5', code='custom'), Message(text='r.F ?{imQNQ5', code='custom')])"
    var_8.__contains__(var_6)

def test_case_43():
    var_0 = None
    var_1 = 'c!t}5"|=vMWW]z'
    var_2 = module_0.Message(text=var_1, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vMWW]z'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == 7078395332879544620
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = [var_2, var_2, var_2]
    var_6 = module_0.ParseError(code=var_0, position=var_0, messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_6.__str__()
    assert var_7 == '{\'\': \'c!t}5"|=vMWW]z\'}'
    var_8 = var_2.__repr__()
    assert var_8 == 'Message(text=\'c!t}5"|=vMWW]z\', code=\'custom\')'
    var_9 = var_4.__repr__()
    assert var_9 == 'ValidationResult(value=None)'

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.BaseError(text=var_2, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = True
    var_5 = module_0.Position(var_3, var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.line_no).__module__}.{type(var_5.line_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.line_no) == 1
    assert f'{type(var_5.column_no).__module__}.{type(var_5.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5.column_no) == 1
    assert var_5.char_index is True
    var_6 = ',N'
    var_7 = module_0.ParseError(text=var_6, code=var_6, key=var_6, position=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_7.keys()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_8) == 1
    var_9 = var_8.__str__()
    assert var_9 == "KeysView(ParseError([Message(text=',N', code=',N', index=[',N'], start_position=Position(line_no=Message(text=None, code='custom'), column_no=Message(text=None, code='custom'), char_index=True), end_position=Position(line_no=Message(text=None, code='custom'), column_no=Message(text=None, code='custom'), char_index=True))]))"
    var_10 = var_7.__eq__(var_0)
    assert var_10 is False
    var_11 = var_1.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_8.__contains__(var_8)

def test_case_45():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.Message(text=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == "Message(text=None, code='custom')"
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"

def test_case_46():
    var_0 = None
    var_1 = 'c!t}5"|=6W]'
    var_2 = module_0.Message(text=var_1, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=6W]'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'i^'
    var_4 = module_0.Message(text=var_3, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'i^'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__hash__()
    assert var_5 == 7078395332879544620
    var_6 = var_2.__repr__()
    assert var_6 == 'Message(text=\'c!t}5"|=6W]\', code=\'custom\')'
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = module_0.Message(text=var_0, key=var_3, position=var_0, start_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == ['i^']
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = -1217
    var_10 = module_0.Position(var_3, var_5, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 'i^'
    assert var_10.column_no == 7078395332879544620
    assert var_10.char_index == -1217
    var_11 = var_10.__eq__(var_0)
    assert var_11 is False
    var_12 = ',N'
    var_13 = var_8.__repr__()
    assert var_13 == "Message(text=None, code='custom', index=['i^'])"
    var_14 = module_0.ParseError(text=var_6, code=var_1, key=var_6, position=var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = module_0.ValidationResult()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is None
    var_16 = var_14.__str__()
    assert var_16 == '{\'Message(text=\\\'c!t}5"|=6W]\\\', code=\\\'custom\\\')\': \'Message(text=\\\'c!t}5"|=6W]\\\', code=\\\'custom\\\')\'}'
    var_17 = var_14.__eq__(var_0)
    assert var_17 is False
    var_18 = var_6.__repr__()
    assert var_18 == '\'Message(text=\\\'c!t}5"|=6W]\\\', code=\\\'custom\\\')\''
    var_19 = var_5.__repr__()
    assert var_19 == '7078395332879544620'
    var_20 = var_12.__repr__()
    assert var_20 == "',N'"
    var_21 = var_14.get(var_0, var_17)
    assert var_21 is False
    var_22 = var_14.items()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_22) == 1
    var_23 = var_22.__contains__(var_7)
    assert var_23 is False
    var_24 = var_14.get(var_14, var_0)

@pytest.mark.xfail(strict=True)
def test_case_47():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.BaseError(text=var_2, code=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = module_0.Position(var_2, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no == "Message(text=None, code='custom')"
    assert f'{type(var_4.column_no).__module__}.{type(var_4.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4.column_no) == 1
    assert f'{type(var_4.char_index).__module__}.{type(var_4.char_index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4.char_index) == 1
    var_5 = ',N'
    var_6 = module_0.ParseError(text=var_5, code=var_5, key=var_5, position=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = var_6.keys()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7) == 1
    var_8 = var_7.__str__()
    assert var_8 == "KeysView(ParseError([Message(text=',N', code=',N', index=[',N'], start_position=Position(line_no=Message(text=None, code='custom'), column_no=Message(text=None, code='custom'), char_index=Message(text=None, code='custom')), end_position=Position(line_no=Message(text=None, code='custom'), column_no=Message(text=None, code='custom'), char_index=Message(text=None, code='custom')))]))"
    var_9 = var_1.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_3.__repr__()
    assert var_10 == 'BaseError(text="Message(text=None, code=\'custom\')", code=\'custom\')'
    var_7.__contains__(var_7)

def test_case_48():
    var_0 = 'dE)'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

@pytest.mark.xfail(strict=True)
def test_case_49():
    var_0 = None
    var_1 = 'c!t}5"|=vW]'
    var_2 = module_0.Message(text=var_1, index=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'c!t}5"|=vW]'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    var_4 = var_2.__repr__()
    assert var_4 == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    var_5 = var_2.__repr__()
    assert var_5 == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    var_6 = 'r.F ?{imQNQ5'
    var_7 = module_0.Message(text=var_5, code=var_0, index=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    assert var_7.code == 'custom'
    assert var_7.index == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_2.__hash__()
    assert var_8 == 7078395332879544620
    var_9 = var_2.__repr__()
    assert var_9 == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    var_10 = module_0.ValidationResult(value=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value == 'Message(text=\'c!t}5"|=vW]\', code=\'custom\')'
    assert var_10.error is None
    var_11 = var_2.__eq__(var_0)
    assert var_11 is False
    var_12 = module_0.Message(text=var_6, position=var_0, start_position=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'r.F ?{imQNQ5'
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = module_0.Position(var_0, var_0, var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no is None
    assert var_13.char_index is False
    var_14 = var_10.__repr__()
    assert var_14 == 'ValidationResult(value=\'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\')'
    var_15 = var_13.__eq__(var_0)
    assert var_15 is False
    var_16 = var_7.__repr__()
    assert var_16 == 'Message(text=\'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\', code=\'custom\', index=\'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\')'
    var_17 = [var_2, var_7, var_12, var_7]
    var_18 = module_0.ParseError(messages=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_18) == 2
    var_19 = module_0.ValidationResult()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_19.value is None
    assert var_19.error is None
    var_20 = var_18.__str__()
    assert var_20 == '{\'\': \'r.F ?{imQNQ5\', \'M\': {\'e\': {\'s\': {\'s\': {\'a\': {\'g\': {\'e\': {\'(\': {\'t\': {\'e\': {\'x\': {\'t\': {\'=\': {"\'": {\'c\': {\'!\': {\'t\': {\'}\': {\'5\': {\'"\': {\'|\': {\'=\': {\'v\': {\'W\': {\']\': {"\'": {\',\': {\' \': {\'c\': {\'o\': {\'d\': {\'e\': {\'=\': {"\'": {\'c\': {\'u\': {\'s\': {\'t\': {\'o\': {\'m\': {"\'": {\')\': \'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\'}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}'
    var_21 = var_18.__eq__(var_19)
    assert var_21 is False
    var_22 = var_7.__repr__()
    assert var_22 == 'Message(text=\'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\', code=\'custom\', index=\'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\')'
    var_23 = var_10.__repr__()
    assert var_23 == 'ValidationResult(value=\'Message(text=\\\'c!t}5"|=vW]\\\', code=\\\'custom\\\')\')'
    var_24 = True
    var_25 = (var_17, var_24, var_0)
    var_26 = var_23.__eq__(var_25)
    var_24.__iter__()