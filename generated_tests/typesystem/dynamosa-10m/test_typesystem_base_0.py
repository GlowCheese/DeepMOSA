# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0


def test_case_0():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_1():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

def test_case_2():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
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
    var_0 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_0)

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
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__hash__()
    assert var_2 == -6983700205098971111

def test_case_8():
    var_0 = 'g6&SZ(4#IO^\x0cmWf`%W'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()

def test_case_9():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True

def test_case_10():
    var_0 = 'g6&SZ(4#IO^\x0cmWf`%W'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.messages(add_prefix=var_0)

def test_case_11():
    var_0 = '7e:,@,/Ax'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_12():
    var_0 = '\\[iv'
    var_1 = module_0.BaseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_13():
    var_0 = 'g6&SZ(4#IO^\x0cmWf`%W'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is False

def test_case_14():
    var_0 = True
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '{True: True}'

def test_case_15():
    var_0 = -3594
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == -3594
    assert var_1.column_no == -3594
    assert var_1.char_index == -3594

def test_case_16():
    var_0 = 'g6&SZ(4#IO^\x0cmWf`%W'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == 'g6&SZ(4#IO^\x0cmWf`%W'

def test_case_17():
    var_0 = '^!R{U!la+3\rsB'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, code=var_0, key=var_0, index=var_0)

def test_case_18():
    var_0 = False
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "ValidationError([Message(text=False, code='custom', index=[False], position=False)])"

def test_case_19():
    var_0 = 'g6&SZ(4#IO^\x0cmWf`%W'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__repr__()
    assert var_2 == "BaseError(text='g6&SZ(4#IO^\\x0cmWf`%W', code='custom')"

def test_case_20():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

def test_case_21():
    var_0 = 'g6&SZ(4#IO^\x0cmWf`%'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__len__()
    assert var_2 == 1

def test_case_22():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_23():
    var_0 = None
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is True
    assert var_2.char_index is True
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

def test_case_24():
    var_0 = False
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = False
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    module_0.ValidationError(text=var_2, code=var_2, position=var_2, messages=var_1)

def test_case_26():
    var_0 = True
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is True
    assert var_1.error is None

def test_case_27():
    var_0 = '7e:,@,/Ax'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.get(var_1)

def test_case_28():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.ValidationResult(error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.Message'
    var_3 = var_2.__repr__()
    assert var_3 == "ValidationResult(error=Message(text=None, code='custom'))"

def test_case_29():
    var_0 = -1772
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == -1772
    assert var_1.column_no == -1772
    assert var_1.char_index == -1772
    var_2 = var_1.__repr__()
    assert var_2 == 'Position(line_no=-1772, column_no=-1772, char_index=-1772)'

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 't!!jH|.R9`'
    module_0.ValidationError(text=var_0, messages=var_0)

def test_case_31():
    var_0 = 't!!jH|.R9`'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, position=var_0, end_position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = False
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    module_0.ValidationError(key=var_0, messages=var_1)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 930
    var_1 = 'J'
    module_0.ValidationError(code=var_1, messages=var_0)

def test_case_34():
    var_0 = False
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.messages()

def test_case_35():
    var_0 = False
    var_1 = module_0.ValidationError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_36():
    var_0 = None
    var_1 = module_0.Message(text=var_0, index=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = module_0.Position(var_0, var_2, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is False
    assert var_3.char_index is False
    var_4 = '\x0bbiJf1]Er\nxJr;Y'
    var_5 = module_0.Message(text=var_4, code=var_4, key=var_0, start_position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text == '\x0bbiJf1]Er\nxJr;Y'
    assert var_5.code == '\x0bbiJf1]Er\nxJr;Y'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert var_5.end_position is None
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text='\\x0bbiJf1]Er\\nxJr;Y', code='\\x0bbiJf1]Er\\nxJr;Y', start_position=Position(line_no=None, column_no=False, char_index=False), end_position=None)"

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = -319
    module_0.ValidationError(position=var_0, messages=var_0)

def test_case_38():
    var_0 = 'x'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, position=var_0, start_position=var_0, end_position=var_0)

def test_case_39():
    var_0 = None
    var_1 = True
    var_2 = module_0.ValidationError(text=var_1, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is None
    assert var_3.column_no is None
    assert var_3.char_index is True
    var_4 = var_2.__contains__(var_0)
    assert var_4 is False
    var_5 = 2164
    var_6 = module_0.Position(var_5, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 2164
    assert var_6.column_no is False
    assert var_6.char_index is False
    var_7 = var_6.__eq__(var_3)
    assert var_7 is False
    var_8 = module_0.Message(text=var_4, code=var_0, index=var_0, position=var_0, start_position=var_5, end_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is False
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position == 2164
    assert var_8.end_position is None

def test_case_40():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_1, error=var_2)

def test_case_41():
    var_0 = None
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__iter__()
    var_3 = module_0.Message(text=var_0, index=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert f'{type(var_3.index).__module__}.{type(var_3.index).__qualname__}' == 'builtins.generator'
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__hash__()
    assert var_4 == 6152802414649224427

def test_case_42():
    var_0 = None
    var_1 = 'tB!!Nj.9`'
    var_2 = module_0.Message(text=var_1, position=var_0, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'tB!!Nj.9`'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.Message'
    var_4 = var_3.__bool__()
    assert var_4 is False
    var_5 = module_0.Position(var_4, var_1, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no is False
    assert var_5.column_no == 'tB!!Nj.9`'
    assert var_5.char_index is None
    var_6 = False
    var_7 = -2740
    var_8 = module_0.Position(var_6, var_0, var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is False
    assert var_8.column_no is None
    assert var_8.char_index == -2740
    var_9 = var_8.__eq__(var_5)
    assert var_9 is False
    var_10 = module_0.Position(var_0, var_9, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is False
    assert var_10.char_index is None
    var_11 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = 384
    var_2 = module_0.ValidationError(text=var_0, code=var_0, key=var_1, position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.ValidationError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_2)
    assert var_4 is False
    var_5 = var_3.__repr__()
    assert var_5 == "ValidationError(text=ValidationError([Message(text=ValidationResult(value=None), code=ValidationResult(value=None), index=[384], position=ValidationResult(value=None))]), code='custom')"
    var_6 = var_2.keys()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6) == 1
    var_7 = var_2.__len__()
    assert var_7 == 1
    var_6.items()

def test_case_44():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = module_0.Message(text=var_0, code=var_1, index=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert f'{type(var_2.code).__module__}.{type(var_2.code).__qualname__}' == 'typesystem.base.Message'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__hash__()
    assert var_4 == -6983700205098971111
    var_5 = var_2.__eq__(var_0)
    assert var_5 is False
    var_6 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None

def test_case_45():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = [var_1, var_1]
    var_3 = module_0.BaseError(messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.__repr__()
    assert var_4 == "BaseError([Message(text=None, code='custom'), Message(text=None, code='custom')])"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_4, messages=var_2)

def test_case_46():
    var_0 = '"EjA+nIs4w'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = [var_2, var_2]
    var_4 = module_0.BaseError(messages=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_4) == 1
    var_5 = var_4.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5) == 1
    var_6 = var_4.__str__()
    assert var_6 == "{'': None}"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_0, messages=var_3)