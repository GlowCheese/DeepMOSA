# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = 'OQyB0G`M 9b'
    var_1 = module_0.BaseError(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_1():
    var_0 = "|c~&cl^:'8|>(xae\x0b"
    var_1 = ''
    var_2 = None
    var_3 = module_0.Message(text=var_0, code=var_1, key=var_2, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == "|c~&cl^:'8|>(xae\x0b"
    assert var_3.code == ''
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None

def test_case_2():
    var_0 = None
    var_1 = module_0.Message(text=var_0, position=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_3():
    with pytest.raises(AssertionError):
        module_0.BaseError()

def test_case_4():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_5():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None

def test_case_6():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.__eq__(var_2)
    assert var_4 is False
    var_5 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is False
    assert var_5.error is None
    var_6 = var_1.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = 392
    var_8 = module_0.Position(var_2, var_2, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert var_8.char_index is False
    var_9 = module_0.Message(text=var_2, key=var_6, start_position=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is None
    assert var_9.code == 'custom'
    assert f'{type(var_9.index).__module__}.{type(var_9.index).__qualname__}' == 'builtins.list'
    assert len(var_9.index) == 1
    assert var_9.start_position is False
    assert var_9.end_position is None
    var_10 = module_0.Position(var_2, var_7, var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no == 392
    assert var_10.char_index == '.s;'
    var_11 = module_0.Message(text=var_6, position=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_11.text).__module__}.{type(var_11.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11.text) == 1
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_11.end_position).__module__}.{type(var_11.end_position).__qualname__}' == 'typesystem.base.Position'
    var_12 = var_11.__repr__()
    assert var_12 == "Message(text=ItemsView(BaseError(text='.s;', code='custom')), code='custom', position=Position(line_no=None, column_no=None, char_index=False))"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_6, key=var_2)

def test_case_7():
    var_0 = 's;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == 's;'

def test_case_8():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '.s;'
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_4, position=var_4, messages=var_4)

def test_case_9():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '.s;'
    var_3 = module_0.Message(text=var_2, index=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '.s;'
    assert var_3.code == 'custom'
    assert f'{type(var_3.index).__module__}.{type(var_3.index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.index) == 1
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='.s;', code='custom', index=BaseError(text='.s;', code='custom'))"

def test_case_10():
    var_0 = ',PU\r?8C7o_'
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ',PU\r?8C7o_'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == ',PU\r?8C7o_'
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=',PU\\r?8C7o_', code='custom', start_position=',PU\\r?8C7o_', end_position=None)"

def test_case_11():
    var_0 = '~se'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == '~se'
    assert var_2.error is None
    var_3 = var_1.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = var_1.__contains__(var_0)
    assert var_5 is False

def test_case_12():
    var_0 = 'cr_+7UUsw1nmw'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.messages()
    var_4 = var_2.__contains__(var_1)
    assert var_4 is False

def test_case_13():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '.s;'
    var_3 = module_0.Message(text=var_2, index=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == '.s;'
    assert var_3.code == 'custom'
    assert f'{type(var_3.index).__module__}.{type(var_3.index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3.index) == 1
    assert var_3.start_position is None
    assert var_3.end_position is None

def test_case_14():
    var_0 = '~se'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == '~se'
    assert var_1.error is None

def test_case_15():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, index=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == '.s;'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_1.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='.s;', code='custom', index=BaseError(text='.s;', code='custom'))"
    var_5 = var_3.__contains__(var_3)
    assert var_5 is False

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = None
    var_2 = ',!U{p(G5n'
    var_3 = None
    var_4 = module_0.Message(text=var_2, code=var_3, start_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == ',!U{p(G5n'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = [var_4, var_4, var_4]
    var_6 = module_0.ValidationError(messages=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = var_6.__contains__(var_1)
    assert var_7 is False
    var_7.__contains__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    module_0.ParseError(position=var_0, messages=var_0)

def test_case_18():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    assert var_2 == 4234175225196340981
    var_3 = var_1.__str__()
    assert var_3 == '.s;'
    var_4 = var_3.__hash__()
    assert var_4 == -5231359117271325037
    var_5 = var_3.__iter__()
    var_6 = var_3.__repr__()
    assert var_6 == "'.s;'"
    var_7 = var_1.__contains__(var_5)
    assert var_7 is False

def test_case_19():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__str__()
    assert var_3 == '.s;'
    var_4 = 'pqgQxK|\x0c'
    var_5 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_5.__bool__()
    assert var_6 is True
    var_7 = module_0.Message(text=var_4, position=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'pqgQxK|\x0c'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = None
    var_9 = var_1.__iter__()
    var_10 = var_8.__repr__()
    assert var_10 == 'None'
    var_11 = var_1.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    var_12 = var_11.__repr__()
    assert var_12 == "ValuesView(BaseError(text='.s;', code='custom'))"

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = '~se'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == '~se'
    assert var_2.error is None
    var_3 = var_1.__iter__()
    var_4 = var_3.__hash__()
    var_5 = var_3.__iter__()
    module_0.ParseError(code=var_5, messages=var_3)

def test_case_21():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == '.s;'
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_1.values()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_4) == 1
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_3, key=var_2, position=var_4, messages=var_4)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = ',PU\r?8C7o_'
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ',PU\r?8C7o_'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == ',PU\r?8C7o_'
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=',PU\\r?8C7o_', code='custom', start_position=',PU\\r?8C7o_', end_position=None)"
    var_6 = module_0.BaseError(text=var_0, code=var_0, key=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.__hash__()
    assert var_7 == -8153505829992923353
    var_8 = var_6.items()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_8) == 1
    var_9 = var_8.__eq__(var_4)
    var_10 = var_6.__str__()
    assert var_10 == "{',PU\\r?8C7o_': ',PU\\r?8C7o_'}"
    var_11 = var_6.__iter__()
    var_11.items()

def test_case_23():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = '.s;'
    var_2 = module_0.BaseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_1, key=var_1, position=var_3, start_position=var_3, end_position=var_3)

def test_case_24():
    var_0 = ',PU\r?8C7o_'
    var_1 = module_0.Message(text=var_0, start_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text == ',PU\r?8C7o_'
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position == ',PU\r?8C7o_'
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=',PU\\r?8C7o_', code='custom', start_position=',PU\\r?8C7o_', end_position=None)"
    var_5 = module_0.Message(text=var_2, code=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = var_1.__repr__()
    assert var_7 == "Message(text=',PU\\r?8C7o_', code='custom', start_position=',PU\\r?8C7o_', end_position=None)"

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = False
    var_1 = ',PU\r?8C7o_'
    var_2 = module_0.Message(text=var_1, start_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == ',PU\r?8C7o_'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position == ',PU\r?8C7o_'
    assert var_2.end_position is None
    var_3 = None
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=',PU\\r?8C7o_', code='custom', start_position=',PU\\r?8C7o_', end_position=None)"
    var_5 = module_0.BaseError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_5) == 1
    var_6 = var_5.__hash__()
    assert var_6 == -8153505829992923353
    var_7 = var_5.__eq__(var_1)
    assert var_7 is False
    var_8 = var_7.__repr__()
    assert var_8 == 'False'
    var_9 = var_5.__repr__()
    assert var_9 == "BaseError([Message(text=False, code=',PU\\r?8C7o_', index=[',PU\\r?8C7o_'])])"
    var_7.__contains__(var_3)

def test_case_26():
    var_0 = None
    var_1 = -479
    var_2 = module_0.Position(var_0, var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no is None
    assert var_2.column_no is None
    assert var_2.char_index == -479
    var_3 = var_2.__eq__(var_0)
    assert var_3 is False

def test_case_27():
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
def test_case_28():
    var_0 = '.s;'
    var_1 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error == '.s;'
    var_2 = var_1.__repr__()
    assert var_2 == "ValidationResult(error='.s;')"
    var_3 = None
    var_4 = []
    module_0.ValidationError(text=var_3, code=var_3, messages=var_4)

def test_case_29():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()
    var_3 = None
    var_4 = var_1.__iter__()
    var_5 = var_4.__eq__(var_3)
    var_6 = var_4.__str__()
    with pytest.raises(AssertionError):
        module_0.Message(text=var_3, code=var_3, key=var_4, index=var_4, position=var_3, end_position=var_4)

def test_case_30():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.__eq__(var_2)
    assert var_4 is False
    var_5 = var_1.__str__()
    assert var_5 == '.s;'
    var_6 = var_1.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = var_1.messages()
    var_8 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is False
    assert var_8.error is None
    var_9 = var_1.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = module_0.Position(var_2, var_2, var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no is None
    assert f'{type(var_10.char_index).__module__}.{type(var_10.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10.char_index) == 1
    var_11 = var_9.__repr__()
    assert var_11 == "ItemsView(BaseError(text='.s;', code='custom'))"
    var_12 = 321
    var_13 = False
    var_14 = module_0.Position(var_4, var_12, var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no is False
    assert var_14.column_no == 321
    assert var_14.char_index is False
    var_15 = var_1.__len__()
    assert var_15 == 1
    var_16 = var_15.__repr__()
    assert var_16 == '1'
    var_17 = 2813
    var_18 = module_0.Position(var_17, var_3, var_9)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Position'
    assert var_18.line_no == 2813
    assert var_18.column_no is False
    assert f'{type(var_18.char_index).__module__}.{type(var_18.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_18.char_index) == 1
    var_19 = var_1.values()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_19) == 1
    var_20 = var_9.__len__()
    assert var_20 == 1
    var_21 = module_0.Message(text=var_15, position=var_10)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text == 1
    assert var_21.code == 'custom'
    assert var_21.index == []
    assert f'{type(var_21.start_position).__module__}.{type(var_21.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_21.end_position).__module__}.{type(var_21.end_position).__qualname__}' == 'typesystem.base.Position'
    var_22 = var_21.__eq__(var_21)
    assert var_22 is True
    var_23 = var_6.__iter__()
    var_24 = var_21.__repr__()
    assert var_24 == "Message(text=1, code='custom', position=Position(line_no=None, column_no=None, char_index=ItemsView(BaseError(text='.s;', code='custom'))))"
    var_25 = []
    with pytest.raises(AssertionError):
        module_0.BaseError(messages=var_25)

def test_case_31():
    var_0 = '$9'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__iter__()
    var_3 = None
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = var_1.__str__()
    assert var_5 == '$9'
    var_6 = var_1.__contains__(var_3)
    assert var_6 is False
    var_7 = var_6.__repr__()
    assert var_7 == 'False'
    var_8 = 'rI,V$0'
    with pytest.raises(AssertionError):
        module_0.Message(text=var_8, key=var_7, position=var_6, end_position=var_6)

def test_case_32():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.__str__()
    assert var_4 == '.s;'
    var_5 = var_1.__iter__()
    var_6 = var_1.messages()
    var_7 = module_0.Message(text=var_3, code=var_0, key=var_4, position=var_2, start_position=var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is False
    assert var_7.code == '.s;'
    assert var_7.index == ['.s;']
    assert var_7.start_position is False
    assert var_7.end_position is None
    var_8 = module_0.Position(var_2, var_2, var_3)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is None
    assert var_8.column_no is None
    assert var_8.char_index is False
    var_9 = var_7.__repr__()
    assert var_9 == "Message(text=False, code='.s;', index=['.s;'], start_position=False, end_position=None)"
    var_10 = module_0.Message(text=var_2, code=var_0, start_position=var_2)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is None
    assert var_10.code == '.s;'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_1.__len__()
    assert var_11 == 1
    var_12 = var_11.__repr__()
    assert var_12 == '1'
    var_13 = '{B5\x0bg[<c'
    var_14 = module_0.Message(text=var_13, end_position=var_8)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == '{B5\x0bg[<c'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert var_14.start_position is None
    assert f'{type(var_14.end_position).__module__}.{type(var_14.end_position).__qualname__}' == 'typesystem.base.Position'
    var_15 = module_0.Message(text=var_11, position=var_8)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text == 1
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.Position'
    var_16 = var_15.__eq__(var_7)
    assert var_16 is False
    var_17 = var_15.__repr__()
    assert var_17 == "Message(text=1, code='custom', position=Position(line_no=None, column_no=None, char_index=False))"
    with pytest.raises(AssertionError):
        module_0.BaseError()

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.values()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_2)
    var_5 = var_1.__hash__()
    assert var_5 == 4234175225196340981
    var_6 = var_1.__str__()
    assert var_6 == '.s;'
    var_7 = var_1.messages(add_prefix=var_3)
    var_8 = module_0.Message(text=var_2, end_position=var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = module_0.Message(text=var_3, end_position=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_9.text).__module__}.{type(var_9.text).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_9.text) == 1
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position == 4234175225196340981
    var_10 = var_9.__eq__(var_7)
    assert var_10 is False
    var_11 = var_8.__repr__()
    assert var_11 == "Message(text=None, code='custom')"
    var_1.__getitem__(var_3)

def test_case_34():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.__eq__(var_2)
    assert var_4 is False
    var_5 = var_1.__str__()
    assert var_5 == '.s;'
    var_6 = var_1.__iter__()
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_1, error=var_3)

def test_case_35():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__str__()
    assert var_4 == '.s;'
    var_5 = var_1.__iter__()
    var_6 = var_1.__iter__()
    var_7 = var_1.messages()
    var_8 = module_0.ValidationResult(value=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == '.s;'
    assert var_8.error is None
    var_9 = var_1.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = var_1.__iter__()
    var_11 = module_0.Message(text=var_9, code=var_0, key=var_4, position=var_2, start_position=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_11.text).__module__}.{type(var_11.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11.text) == 1
    assert var_11.code == '.s;'
    assert var_11.index == ['.s;']
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11.start_position) == 1
    assert var_11.end_position is None
    var_12 = module_0.Position(var_2, var_2, var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is None
    assert var_12.column_no is None
    assert f'{type(var_12.char_index).__module__}.{type(var_12.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12.char_index) == 1
    var_13 = var_11.__repr__()
    assert var_13 == "Message(text=ItemsView(BaseError(text='.s;', code='custom')), code='.s;', index=['.s;'], start_position=ItemsView(BaseError(text='.s;', code='custom')), end_position=None)"
    var_14 = 321
    var_15 = False
    var_16 = module_0.Position(var_3, var_14, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no is False
    assert var_16.column_no == 321
    assert var_16.char_index is False
    var_17 = module_0.Message(text=var_2, code=var_0, start_position=var_2)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text is None
    assert var_17.code == '.s;'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = var_0.__repr__()
    assert var_18 == "'.s;'"
    var_19 = module_0.Position(var_3, var_1, var_9)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Position'
    assert var_19.line_no is False
    assert f'{type(var_19.column_no).__module__}.{type(var_19.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_19.column_no) == 1
    assert f'{type(var_19.char_index).__module__}.{type(var_19.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_19.char_index) == 1
    var_20 = var_1.values()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_20) == 1
    var_21 = module_0.Message(text=var_9, position=var_12)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_21.text).__module__}.{type(var_21.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_21.text) == 1
    assert var_21.code == 'custom'
    assert var_21.index == []
    assert f'{type(var_21.start_position).__module__}.{type(var_21.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_21.end_position).__module__}.{type(var_21.end_position).__qualname__}' == 'typesystem.base.Position'
    var_22 = var_21.__eq__(var_11)
    assert var_22 is False
    var_23 = var_21.__repr__()
    assert var_23 == "Message(text=ItemsView(BaseError(text='.s;', code='custom')), code='custom', position=Position(line_no=None, column_no=None, char_index=ItemsView(BaseError(text='.s;', code='custom'))))"

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__contains__(var_2)
    assert var_3 is False
    var_4 = var_1.__eq__(var_2)
    assert var_4 is False
    var_5 = var_1.__str__()
    assert var_5 == '.s;'
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = var_1.__iter__()
    var_8 = var_1.messages()
    var_9 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is False
    assert var_9.error is None
    var_10 = var_1.items()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10) == 1
    var_11 = module_0.Message(text=var_1, index=var_3, position=var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_11.text).__module__}.{type(var_11.text).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11.text) == 1
    assert var_11.code == 'custom'
    assert var_11.index is False
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = -2714
    var_13 = module_0.Position(var_4, var_10, var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is False
    assert f'{type(var_13.column_no).__module__}.{type(var_13.column_no).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_13.column_no) == 1
    assert var_13.char_index == -2714
    var_14 = var_10.__contains__(var_9)
    assert var_14 is False
    var_15 = var_14.__repr__()
    assert var_15 == 'False'
    var_10.get(var_14, var_2)

def test_case_37():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__str__()
    assert var_4 == '.s;'
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_1.__iter__()
    var_7 = var_1.messages()
    var_8 = module_0.ValidationResult(value=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == '.s;'
    assert var_8.error is None
    var_9 = var_1.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = module_0.Message(text=var_9, code=var_0, key=var_4, position=var_2, start_position=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_10.text).__module__}.{type(var_10.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10.text) == 1
    assert var_10.code == '.s;'
    assert var_10.index == ['.s;']
    assert f'{type(var_10.start_position).__module__}.{type(var_10.start_position).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_10.start_position) == 1
    assert var_10.end_position is None
    var_11 = module_0.Position(var_2, var_2, var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Position'
    assert var_11.line_no is None
    assert var_11.column_no is None
    assert f'{type(var_11.char_index).__module__}.{type(var_11.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11.char_index) == 1
    var_12 = var_10.__repr__()
    assert var_12 == "Message(text=ItemsView(BaseError(text='.s;', code='custom')), code='.s;', index=['.s;'], start_position=ItemsView(BaseError(text='.s;', code='custom')), end_position=None)"
    var_13 = 321
    var_14 = False
    var_15 = module_0.Position(var_3, var_13, var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no is False
    assert var_15.column_no == 321
    assert var_15.char_index is False
    var_16 = module_0.Message(text=var_2, code=var_0, start_position=var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == '.s;'
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = var_0.__repr__()
    assert var_17 == "'.s;'"
    var_18 = None
    var_19 = 2813
    var_20 = module_0.Position(var_19, var_1, var_9)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no == 2813
    assert f'{type(var_20.column_no).__module__}.{type(var_20.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_20.column_no) == 1
    assert f'{type(var_20.char_index).__module__}.{type(var_20.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_20.char_index) == 1
    var_21 = var_1.values()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_21) == 1
    var_22 = var_9.__len__()
    assert var_22 == 1
    var_23 = module_0.Message(text=var_18, index=var_21, start_position=var_15, end_position=var_11)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text is None
    assert var_23.code == 'custom'
    assert f'{type(var_23.index).__module__}.{type(var_23.index).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_23.index) == 1
    assert f'{type(var_23.start_position).__module__}.{type(var_23.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_23.end_position).__module__}.{type(var_23.end_position).__qualname__}' == 'typesystem.base.Position'
    var_24 = var_10.__eq__(var_9)
    assert var_24 is False
    var_25 = var_23.__repr__()
    assert var_25 == "Message(text=None, code='custom', index=ValuesView(BaseError(text='.s;', code='custom')), start_position=Position(line_no=False, column_no=321, char_index=False), end_position=Position(line_no=None, column_no=None, char_index=ItemsView(BaseError(text='.s;', code='custom'))))"
    with pytest.raises(AssertionError):
        module_0.BaseError(code=var_4)

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__str__()
    assert var_3 == '.s;'
    var_4 = var_1.__len__()
    assert var_4 == 1
    var_5 = var_1.__iter__()
    var_6 = var_1.messages()
    var_7 = module_0.ValidationError(text=var_3, code=var_2, key=var_4, messages=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = var_4.__repr__()
    assert var_8 == '1'
    var_9 = 321
    var_10 = module_0.Position(var_4, var_9, var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no == 1
    assert var_10.column_no == 321
    assert var_10.char_index == 1
    var_11 = var_0.__repr__()
    assert var_11 == "'.s;'"
    var_12 = module_0.Position(var_0, var_1, var_7)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no == '.s;'
    assert f'{type(var_12.column_no).__module__}.{type(var_12.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12.column_no) == 1
    assert f'{type(var_12.char_index).__module__}.{type(var_12.char_index).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12.char_index) == 1
    var_13 = var_1.values()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_13) == 1
    var_14 = var_7.__len__()
    assert var_14 == 1
    var_15 = module_0.Message(text=var_7, position=var_1)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_15.text).__module__}.{type(var_15.text).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15.text) == 1
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15.start_position) == 1
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_15.end_position) == 1
    var_16 = var_15.__eq__(var_15)
    assert var_16 is False
    var_17 = var_15.__repr__()
    assert var_17 == "Message(text=ValidationError([Message(text='.s;', code='custom', index=[1])]), code='custom', start_position=BaseError(text='.s;', code='custom'), end_position=BaseError(text='.s;', code='custom'))"
    module_0.BaseError(key=var_2, messages=var_7)

def test_case_39():
    var_0 = '.s;'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__str__()
    assert var_4 == '.s;'
    var_5 = var_1.__iter__()
    var_6 = var_1.__iter__()
    var_7 = var_1.messages()
    var_8 = module_0.ValidationResult(value=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == '.s;'
    assert var_8.error is None
    var_9 = var_1.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    var_10 = var_5.__repr__()
    var_11 = 321
    var_12 = module_0.Position(var_3, var_11, var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Position'
    assert var_12.line_no is False
    assert var_12.column_no == 321
    assert var_12.char_index is False
    var_13 = module_0.Message(text=var_2, code=var_5, index=var_2, position=var_12, start_position=var_2)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Message'
    assert var_13.text is None
    assert f'{type(var_13.code).__module__}.{type(var_13.code).__qualname__}' == 'builtins.dict_keyiterator'
    assert var_13.index == []
    assert f'{type(var_13.start_position).__module__}.{type(var_13.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_13.end_position).__module__}.{type(var_13.end_position).__qualname__}' == 'typesystem.base.Position'
    var_14 = module_0.Position(var_3, var_1, var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Position'
    assert var_14.line_no is False
    assert f'{type(var_14.column_no).__module__}.{type(var_14.column_no).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_14.column_no) == 1
    assert f'{type(var_14.char_index).__module__}.{type(var_14.char_index).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14.char_index) == 1
    var_15 = var_1.values()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_15) == 1
    var_16 = module_0.Message(text=var_9, position=var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_16.text).__module__}.{type(var_16.text).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16.text) == 1
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert f'{type(var_16.start_position).__module__}.{type(var_16.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_16.end_position).__module__}.{type(var_16.end_position).__qualname__}' == 'typesystem.base.Position'
    var_17 = var_16.__eq__(var_16)
    assert var_17 is False
    var_18 = var_16.__repr__()
    assert var_18 == "Message(text=ItemsView(BaseError(text='.s;', code='custom')), code='custom', start_position=Position(line_no=False, column_no=.s;, char_index=ItemsView(BaseError(text='.s;', code='custom'))), end_position=Position(line_no=False, column_no=.s;, char_index=ItemsView(BaseError(text='.s;', code='custom'))))"

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = repr(var_2)
    assert var_3 == "ValidationError(text='Invalid value', code='invalid')"
    var_4 = 'Field required'
    var_5 = 'required'
    var_6 = 'username'
    var_7 = module_0.ValidationError(text=var_4, code=var_5, key=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7) == 1
    var_8 = repr(var_7)
    assert var_8 == "ValidationError([Message(text='Field required', code='required', index=['username'])])"
    var_9 = 'Too short'
    var_10 = [var_6]
    var_11 = module_0.Message(text=var_9, code=var_5, index=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text == 'Too short'
    assert var_11.code == 'required'
    assert var_11.index == ['username']
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = 'Invalid format'
    var_13 = 'format'
    var_14 = 'email'
    var_15 = [var_14]
    var_16 = module_0.Message(text=var_12, code=var_13, index=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'Invalid format'
    assert var_16.code == 'format'
    assert var_16.index == ['email']
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = [var_11, var_16]
    var_18 = module_0.ValidationError(messages=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_18) == 2
    var_19 = repr(var_18)
    var_20 = 1
    var_21 = 5
    var_22 = 4
    var_23 = module_0.Position(var_20, var_21, var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Position'
    assert var_23.line_no == 1
    assert var_23.column_no == 5
    assert var_23.char_index == 4
    var_24 = 'SynHax error'
    var_25 = ',<'
    var_26 = module_0.ValidationError(text=var_24, code=var_25, position=var_23)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_26) == 1
    var_27 = repr(var_26)
    var_28 = 'Invalid JSON'
    var_29 = 'invalid_json'
    var_30 = module_0.ParseError(text=var_28, code=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_30) == 1
    var_31 = repr(var_30)
    assert var_31 == "ParseError(text='Invalid JSON', code='invalid_json')"
    var_32 = 'Invalid'
    var_33 = 'users'
    var_34 = 0
    var_35 = 'name'
    var_36 = [var_33, var_34, var_35]
    var_37 = module_0.Message(text=var_32, code=var_1, index=var_36)
    assert f'{type(var_37).__module__}.{type(var_37).__qualname__}' == 'typesystem.base.Message'
    assert var_37.text == 'Invalid'
    assert var_37.code == 'invalid'
    assert var_37.index == ['users', 0, 'name']
    assert var_37.start_position is None
    assert var_37.end_position is None
    var_38 = [var_37]
    var_39 = module_0.ValidationError(messages=var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_39) == 1
    var_40 = repr(var_39)
    var_41 = []
    module_0.ValidationError(messages=var_41)

def test_case_41():
    var_0 = 'Invalid value'
    var_1 = 'invalid'
    var_2 = module_0.ValidationError(text=var_0, code=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = str(var_2)
    assert var_3 == 'Invalid value'
    var_4 = 'Field required'
    var_5 = 'username'
    var_6 = module_0.ValidationError(text=var_4, code=var_5, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6) == 1
    var_7 = str(var_6)
    assert var_7 == "{'username': 'Field required'}"
    var_8 = [var_5]
    var_9 = module_0.Message(text=var_4, code=var_1, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Field required'
    assert var_9.code == 'invalid'
    assert var_9.index == ['username']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = 'Too short'
    var_11 = 'min_length'
    var_12 = 'password'
    var_13 = [var_12]
    var_14 = module_0.Message(text=var_10, code=var_11, index=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'Too short'
    assert var_14.code == 'min_length'
    assert var_14.index == ['password']
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = [var_9, var_14]
    var_16 = module_0.ValidationError(messages=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_16) == 2
    var_17 = str(var_16)
    assert var_17 == "{'username': 'Field required', 'password': 'Too short'}"
    var_18 = 'Invalid email'
    var_19 = 'users'
    var_20 = 0
    var_21 = 'email'
    var_22 = [var_19, var_20, var_21]
    var_23 = module_0.Message(text=var_18, code=var_1, index=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text == 'Invalid email'
    assert var_23.code == 'invalid'
    assert var_23.index == ['users', 0, 'email']
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = 'max_length'
    var_25 = 1
    var_26 = 'name'
    var_27 = [var_19, var_25, var_26]
    var_28 = module_0.Message(text=var_21, code=var_24, index=var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Message'
    assert var_28.text == 'email'
    assert var_28.code == 'max_length'
    assert var_28.index == ['users', 1, 'name']
    assert var_28.start_position is None
    assert var_28.end_position is None
    var_29 = [var_23, var_28]
    var_30 = module_0.ValidationError(messages=var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_30) == 1
    var_31 = str(var_30)
    var_32 = 'Invalid format'
    var_33 = []
    var_34 = module_0.Message(text=var_32, code=var_1, index=var_33)
    assert f'{type(var_34).__module__}.{type(var_34).__qualname__}' == 'typesystem.base.Message'
    assert var_34.text == 'Invalid format'
    assert var_34.code == 'invalid'
    assert var_34.index == []
    assert var_34.start_position is None
    assert var_34.end_position is None
    var_35 = 'Missing field'
    var_36 = 'profile'
    var_37 = 'age'
    var_38 = [var_36, var_37]
    var_39 = module_0.Message(text=var_35, code=var_36, index=var_38)
    assert f'{type(var_39).__module__}.{type(var_39).__qualname__}' == 'typesystem.base.Message'
    assert var_39.text == 'Missing field'
    assert var_39.code == 'profile'
    assert var_39.index == ['profile', 'age']
    assert var_39.start_position is None
    assert var_39.end_position is None
    var_40 = [var_34, var_39]
    var_41 = module_0.ValidationError(messages=var_40)
    assert f'{type(var_41).__module__}.{type(var_41).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_41) == 2
    var_42 = str(var_41)
    assert var_42 == "{'': 'Invalid format', 'profile': {'age': 'Missing field'}}"
    var_43 = 'Invalid JSON'
    var_44 = 'parse_error'
    var_45 = module_0.ParseError(text=var_43, code=var_44)
    assert f'{type(var_45).__module__}.{type(var_45).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_45) == 1
    var_46 = str(var_45)
    assert var_46 == 'Invalid JSON'
    var_47 = 5
    var_48 = 4
    var_49 = module_0.Position(var_25, var_47, var_48)
    assert f'{type(var_49).__module__}.{type(var_49).__qualname__}' == 'typesystem.base.Position'
    assert var_49.line_no == 1
    assert var_49.column_no == 5
    assert var_49.char_index == 4
    var_50 = 'Syntax error'
    var_51 = 'syntax'
    var_52 = 'query'
    var_53 = module_0.ValidationError(text=var_50, code=var_51, key=var_52, position=var_49)
    assert f'{type(var_53).__module__}.{type(var_53).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_53) == 1
    var_54 = str(var_53)
    assert var_54 == "{'query': 'Syntax error'}"