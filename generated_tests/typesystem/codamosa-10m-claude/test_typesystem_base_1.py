# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = 'RzX.cOs~"/FU"\'u\x0bi=7'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, index=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'RzX.cOs~"/FU"\'u\x0bi=7'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    module_0.ValidationError(position=var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = 'Hd\x0b{<U#%!L2_M_lR4Tr'
    var_2 = module_0.Message(text=var_0, code=var_1, index=var_1, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'Hd\x0b{<U#%!L2_M_lR4Tr'
    assert var_2.index == 'Hd\x0b{<U#%!L2_M_lR4Tr'
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text=None, code='Hd\\x0b{<U#%!L2_M_lR4Tr', index='Hd\\x0b{<U#%!L2_M_lR4Tr')"
    var_4 = None
    module_0.ParseError(code=var_4)

def test_case_2():
    var_0 = 'A8+C<iw'
    var_1 = module_0.ParseError(text=var_0, key=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
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
    var_4 = None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = None
    var_7 = [var_1]
    var_8 = module_0.ParseError(position=var_6, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_1.__eq__(var_0)
    assert var_9 is False
    module_0.ValidationError(position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = 'MNkFE|GnTX3'
    module_0.ValidationError(position=var_0)

def test_case_5():
    var_0 = 'Hd\x0b{<U#%!L2_M_lR4Tr'
    var_1 = None
    var_2 = module_0.Message(text=var_1, code=var_0, index=var_0, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is None
    assert var_2.code == 'Hd\x0b{<U#%!L2_M_lR4Tr'
    assert var_2.index == 'Hd\x0b{<U#%!L2_M_lR4Tr'
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = [var_2, var_2, var_2]
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text=None, code='Hd\\x0b{<U#%!L2_M_lR4Tr', index='Hd\\x0b{<U#%!L2_M_lR4Tr')"
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_0, messages=var_3)

def test_case_6():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = module_0.ParseError(text=var_2, messages=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__eq__(var_1)
    assert var_4 is False
    var_5 = var_0.__repr__()
    assert var_5 == 'None'

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
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
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = None
    var_6 = []
    module_0.ParseError(position=var_5, messages=var_6)

def test_case_8():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None

def test_case_9():
    var_0 = -3260
    var_1 = None
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == -3260
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__eq__(var_1)
    assert var_3 is False

def test_case_10():
    var_0 = 1394
    var_1 = None
    var_2 = False
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == 1394
    assert var_3.column_no is None
    assert var_3.char_index is False
    var_4 = var_3.__repr__()
    assert var_4 == 'Position(line_no=1394, column_no=None, char_index=False)'

def test_case_11():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = var_1.__hash__()
    assert var_3 == 5808716241442466140
    var_4 = None
    var_5 = module_0.Message(text=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'o*/4'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_1, key=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_1.__eq__(var_1)
    assert var_3 is True
    var_4 = var_2.__contains__(var_3)
    assert var_4 is False
    var_4.__len__()

def test_case_13():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False
    var_3 = '_D\x0bll]*NCB@J@l^\x0c5g4'
    var_4 = module_0.Message(text=var_3, key=var_2, index=var_0, end_position=var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == '_D\x0bll]*NCB@J@l^\x0c5g4'
    assert var_4.code == 'custom'
    assert var_4.index == [False]
    assert var_4.start_position is None
    assert var_4.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
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
    var_4 = None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = None
    var_7 = [var_1]
    var_8 = module_0.ParseError(position=var_6, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.__iter__()
    module_0.ValidationError(position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_3 = var_1.__repr__()
    assert var_3 == "Message(text=None, code='custom')"
    var_4 = None
    module_0.ValidationError(position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = None
    var_6 = None
    var_7 = module_0.ParseError(text=var_4, messages=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = var_1.__eq__(var_2)
    assert var_8 is False
    var_9 = var_7.__len__()
    assert var_9 == 1
    var_10 = var_7.__eq__(var_5)
    assert var_10 is False
    var_7.__getitem__(var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = 'o*/4'
    var_2 = module_0.ValidationError(text=var_1, code=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.__contains__(var_0)
    assert var_3 is False
    var_4 = var_2.messages()
    var_3.values()

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = var_1.__hash__()
    assert var_3 == 5808716241442466140
    var_4 = None
    var_5 = module_0.ValidationResult(value=var_4, error=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = None
    var_7 = var_5.__bool__()
    assert var_7 is True
    module_0.ValidationError(code=var_6, messages=var_2)

def test_case_19():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = var_1.__repr__()
    assert var_2 == 'ValidationResult(value=None)'

def test_case_20():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'P(S>DG3fK\\'
    var_3 = module_0.ValidationResult(value=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == 'P(S>DG3fK\\'
    assert var_3.error is None

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    module_0.ValidationError(position=var_0)

def test_case_22():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
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
    var_3 = None
    var_4 = var_2.__iter__()
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = None
    var_7 = None
    var_8 = module_0.ValidationError(text=var_5, key=var_3, position=var_7, messages=var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_8) == 1
    var_9 = var_8.__iter__()
    var_10 = 'J'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_10, key=var_0, position=var_0, messages=var_9)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.ParseError(text=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_1.__eq__(var_2)
    assert var_6 is False
    var_7 = var_5.__eq__(var_3)
    assert var_7 is False
    var_8 = var_5.values()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_8) == 1
    var_9 = var_8.__repr__()
    assert var_9 == 'ValuesView(ParseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = 'Hd\x0b{<U#%!L2_M_lR4Tr'
    var_2 = None
    var_3 = module_0.Message(text=var_2, code=var_1, index=var_1, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'Hd\x0b{<U#%!L2_M_lR4Tr'
    assert var_3.index == 'Hd\x0b{<U#%!L2_M_lR4Tr'
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = [var_3, var_3, var_3]
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text=None, code='Hd\\x0b{<U#%!L2_M_lR4Tr', index='Hd\\x0b{<U#%!L2_M_lR4Tr')"
    var_6 = module_0.BaseError(text=var_0, code=var_2, key=var_2, messages=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_6) == 1
    var_7 = var_6.get(var_0)
    var_8 = None
    module_0.ParseError(code=var_0, key=var_8, position=var_8)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = None
    var_4 = var_1.__repr__()
    assert var_4 == "Message(text=None, code='custom')"
    var_5 = module_0.ParseError(text=var_4, messages=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_1.__eq__(var_2)
    assert var_6 is False
    var_7 = var_5.__eq__(var_3)
    assert var_7 is False
    var_8 = var_5.__str__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = module_0.Position(var_0, var_0, var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no is None
    assert var_9.column_no is None
    assert var_9.char_index == "Message(text=None, code='custom')"
    var_10 = var_9.__repr__()
    assert var_10 == "Position(line_no=None, column_no=None, char_index=Message(text=None, code='custom'))"
    module_0.ParseError(text=var_0, position=var_2)

def test_case_26():
    var_0 = 'P(S>DG3fK\\'
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_2 = var_1.__hash__()
    assert var_2 == 4152311054051068374
    var_3 = None
    var_4 = var_1.__eq__(var_3)
    assert var_4 is False
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_5.__repr__()
    assert var_6 == 'ValidationResult(value=None)'
    var_7 = var_1.__repr__()
    assert var_7 == "ParseError(text='P(S>DG3fK\\\\', code='custom')"
    var_8 = var_1.__contains__(var_3)
    assert var_8 is False
    var_9 = var_1.__str__()
    assert var_9 == 'P(S>DG3fK\\'

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
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
    var_4 = None
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = None
    var_7 = None
    var_8 = module_0.ParseError(text=var_3, code=var_4, key=var_5, position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.__iter__()
    var_10 = var_8.__eq__(var_3)
    assert var_10 is False
    var_11 = var_8.values()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_11) == 1
    var_12 = var_8.__str__()
    assert var_12 == '{"Message(text=None, code=\'custom\')": True}'
    var_11.get(var_9, var_6)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'P(S>DG3fK\\'
    var_3 = module_0.ParseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__hash__()
    assert var_4 == 4152311054051068374
    var_5 = var_1.__eq__(var_0)
    assert var_5 is False
    var_6 = None
    var_7 = var_3.__eq__(var_6)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_1.__repr__()
    assert var_9 == "Message(text=None, code='custom')"
    var_10 = var_3.__repr__()
    assert var_10 == "ParseError(text='P(S>DG3fK\\\\', code='custom')"
    var_11 = var_1.__hash__()
    assert var_11 == 5808716241442466140
    var_12 = module_0.ValidationError(text=var_9, key=var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12) == 1
    var_13 = var_3.__contains__(var_0)
    assert var_13 is False
    var_14 = var_3.__str__()
    assert var_14 == 'P(S>DG3fK\\'
    var_15 = var_13.__str__()
    assert var_15 == 'False'
    var_16 = var_3.__repr__()
    assert var_16 == "ParseError(text='P(S>DG3fK\\\\', code='custom')"
    var_17 = var_12.values()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_17) == 1
    var_18 = var_17.__repr__()
    assert var_18 == 'ValuesView(ValidationError([Message(text="Message(text=None, code=\'custom\')", code=\'custom\', index=[False])]))'
    module_0.ParseError(key=var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'P(S>DG3fK\\'
    var_3 = module_0.ParseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = var_3.__hash__()
    assert var_4 == 4152311054051068374
    var_5 = var_1.__eq__(var_0)
    assert var_5 is False
    var_6 = None
    var_7 = var_3.__eq__(var_6)
    assert var_7 is False
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_7.__repr__()
    assert var_9 == 'False'
    var_10 = var_7.__str__()
    assert var_10 == 'False'
    var_11 = var_3.__repr__()
    assert var_11 == "ParseError(text='P(S>DG3fK\\\\', code='custom')"
    var_12 = var_1.__hash__()
    assert var_12 == 5808716241442466140
    var_13 = None
    var_14 = module_0.ValidationError(text=var_9, key=var_5)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_14) == 1
    var_15 = var_3.__contains__(var_0)
    assert var_15 is False
    var_16 = var_3.__str__()
    assert var_16 == 'P(S>DG3fK\\'
    var_17 = var_15.__str__()
    assert var_17 == 'False'
    var_18 = var_7.__hash__()
    assert var_18 == 0
    var_19 = var_3.__repr__()
    assert var_19 == "ParseError(text='P(S>DG3fK\\\\', code='custom')"
    var_20 = var_14.values()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_20) == 1
    module_0.ParseError(key=var_13, position=var_15, messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_30():
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
    var_3 = None
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__eq__(var_4)
    assert var_5 is False
    var_6 = var_4.__iter__()
    var_7 = var_1.__eq__(var_3)
    assert var_7 is False
    var_8 = var_1.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = var_1.__hash__()
    assert var_9 == 5808716241442466140
    var_10 = var_4.__iter__()
    var_11 = None
    var_12 = module_0.ParseError(text=var_8, messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_12.__iter__()
    var_14 = var_12.__eq__(var_2)
    assert var_14 is False
    var_15 = var_12.messages()
    var_16 = var_12.values()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_16) == 1
    var_17 = var_12.__str__()
    assert var_17 == "Message(text=None, code='custom')"
    var_18 = var_1.__hash__()
    assert var_18 == 5808716241442466140
    var_19 = var_16.__repr__()
    assert var_19 == 'ValuesView(ParseError(text="Message(text=None, code=\'custom\')", code=\'custom\'))'
    var_20 = var_13.__str__()
    module_0.ParseError(text=var_0, code=var_13, position=var_11, messages=var_13)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = None
    var_3 = module_0.Position(var_1, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is None
    assert var_3.char_index is True
    var_4 = None
    var_5 = module_0.Message(text=var_2, code=var_4, position=var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert f'{type(var_5.start_position).__module__}.{type(var_5.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_5.end_position).__module__}.{type(var_5.end_position).__qualname__}' == 'typesystem.base.Position'
    var_6 = var_5.__repr__()
    assert var_6 == "Message(text=None, code='custom', position=Position(line_no=True, column_no=None, char_index=True))"
    module_0.ParseError(code=var_4)

def test_case_32():
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
    var_3 = module_0.ValidationResult()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert var_3.error is None
    var_4 = var_3.__iter__()
    var_5 = var_3.__repr__()
    assert var_5 == 'ValidationResult(value=None)'
    var_6 = var_3.__iter__()
    var_7 = None
    var_8 = module_0.ParseError(text=var_5, messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_8.__iter__()
    var_10 = var_8.__eq__(var_2)
    assert var_10 is False
    var_11 = module_0.Message(text=var_8, start_position=var_9, end_position=var_2)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_11.text).__module__}.{type(var_11.text).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11.text) == 1
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert f'{type(var_11.start_position).__module__}.{type(var_11.start_position).__qualname__}' == 'builtins.dict_keyiterator'
    assert var_11.end_position is True
    var_12 = var_11.__str__()
    var_13 = var_11.__hash__()
    assert var_13 == 5808716241442466140
    var_14 = var_9.__repr__()
    var_15 = var_11.__eq__(var_11)
    assert var_15 is False

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'P(S>DG3fK\\'
    var_3 = module_0.ParseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_3) == 1
    var_4 = None
    var_5 = module_0.ValidationResult(error=var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error == 'P(S>DG3fK\\'
    var_6 = var_1.__eq__(var_4)
    assert var_6 is False
    var_7 = var_3.__iter__()
    var_8 = var_1.__eq__(var_0)
    assert var_8 is False
    var_9 = var_5.__repr__()
    assert var_9 == "ValidationResult(error='P(S>DG3fK\\\\')"
    var_10 = None
    var_11 = var_5.__iter__()
    var_12 = module_0.Message(text=var_4, position=var_10, start_position=var_0, end_position=var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text is None
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = var_12.__eq__(var_0)
    assert var_13 is False
    module_0.ParseError(messages=var_10)

def test_case_34():
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
    var_3 = module_0.Position(var_2, var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is None
    assert var_3.char_index is True
    var_4 = None
    var_5 = var_1.__hash__()
    assert var_5 == 5808716241442466140
    var_6 = None
    var_7 = var_1.__eq__(var_4)
    assert var_7 is False
    var_8 = None
    var_9 = module_0.Message(text=var_8, code=var_6, position=var_3)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is None
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert f'{type(var_9.start_position).__module__}.{type(var_9.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_9.end_position).__module__}.{type(var_9.end_position).__qualname__}' == 'typesystem.base.Position'

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_3 = None
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = None
    var_6 = var_1.__eq__(var_3)
    assert var_6 is False
    var_7 = var_4.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = None
    var_10 = var_4.__iter__()
    var_11 = var_4.__iter__()
    var_12 = None
    var_13 = [var_1, var_1, var_1, var_1]
    var_14 = module_0.ParseError(code=var_9, messages=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.__hash__()
    assert var_15 == 6690227223704023551
    var_16 = var_14.__eq__(var_5)
    assert var_16 is False
    var_17 = var_7.__str__()
    assert var_17 == 'ValidationResult(value=None)'
    var_18 = var_14.__str__()
    assert var_18 == "{'': None}"
    var_19 = var_1.__hash__()
    assert var_19 == 5808716241442466140
    var_20 = var_14.__eq__(var_0)
    assert var_20 is False
    var_21 = var_14.__repr__()
    assert var_21 == "ParseError([Message(text=None, code='custom'), Message(text=None, code='custom'), Message(text=None, code='custom'), Message(text=None, code='custom')])"
    module_0.ParseError(code=var_7, key=var_12)

def test_case_36():
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
    var_3 = None
    var_4 = module_0.ValidationResult()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None
    var_5 = var_1.__eq__(var_4)
    assert var_5 is False
    var_6 = None
    var_7 = var_4.__iter__()
    var_8 = var_4.__repr__()
    assert var_8 == 'ValidationResult(value=None)'
    var_9 = var_1.__eq__(var_3)
    assert var_9 is False
    var_10 = var_1.__repr__()
    assert var_10 == "Message(text=None, code='custom')"
    var_11 = 'm7\x0cq'
    var_12 = module_0.Message(text=var_11, key=var_5, start_position=var_3, end_position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text == 'm7\x0cq'
    assert var_12.code == 'custom'
    assert var_12.index == [False]
    assert var_12.start_position is None
    assert var_12.end_position == 'm7\x0cq'
    var_13 = var_4.__repr__()
    assert var_13 == 'ValidationResult(value=None)'
    var_14 = var_1.__eq__(var_7)
    assert var_14 is False
    var_15 = var_4.__iter__()
    var_16 = None
    var_17 = module_0.ParseError(text=var_10, messages=var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_17) == 1
    var_18 = var_17.__iter__()
    var_19 = var_17.__eq__(var_2)
    assert var_19 is False
    var_20 = var_17.values()
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_20) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_10, code=var_10, key=var_9, index=var_20, end_position=var_6)

def test_case_37():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = module_0.Message(text=var_0, code=var_1, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'Error 1'
    assert var_2.code == 'code1'
    assert var_2.index == ['Error 1']
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = 'Error 2'
    var_4 = 'code2'
    var_5 = 'field2'
    var_6 = module_0.Message(text=var_3, code=var_4, key=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 2'
    assert var_6.code == 'code2'
    assert var_6.index == ['field2']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = [var_2, var_6]
    var_8 = module_0.BaseError(messages=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_8) == 2
    var_9 = var_8.messages()
    var_10 = len(var_9)
    assert var_10 == 2
    var_11 = 0
    var_12 = var_8.messages(add_prefix=var_11)
    var_13 = len(var_12)
    assert var_13 == 2
    var_14 = 'users'
    var_15 = var_8.messages(add_prefix=var_14)
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = 'Nested error'
    var_18 = 'nested'
    var_19 = 'name'
    var_20 = [var_14, var_11, var_19]
    var_21 = module_0.Message(text=var_17, code=var_18, index=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.base.Message'
    assert var_21.text == 'Nested error'
    assert var_21.code == 'nested'
    assert var_21.index == ['users', 0, 'name']
    assert var_21.start_position is None
    assert var_21.end_position is None
    var_22 = [var_21]
    var_23 = module_0.BaseError(messages=var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_23) == 1
    var_24 = 'data'
    var_25 = var_23.messages(add_prefix=var_24)
    var_26 = len(var_25)
    assert var_26 == 1
    var_27 = var_8.messages()
    var_28 = var_8.messages()
    var_29 = 'Root error'
    var_30 = 'root'
    var_31 = module_0.Message(text=var_29, code=var_30)
    assert f'{type(var_31).__module__}.{type(var_31).__qualname__}' == 'typesystem.base.Message'
    assert var_31.text == 'Root error'
    assert var_31.code == 'root'
    assert var_31.index == []
    assert var_31.start_position is None
    assert var_31.end_position is None
    var_32 = [var_31]
    var_33 = module_0.BaseError(messages=var_32)
    assert f'{type(var_33).__module__}.{type(var_33).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_33) == 1
    var_34 = var_33.messages()
    var_35 = len(var_34)
    assert var_35 == 1
    var_36 = var_33.messages(add_prefix=var_30)

def test_case_38():
    var_0 = 'test_value'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'test_value'
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
    var_5 = 'keY'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.ValidationResult(value=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == {'keY': 'value'}
    assert var_8.error is None
    var_9 = -13
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = module_0.ValidationResult(value=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value == [-13, 2, 3]
    assert var_13.error is None
    var_14 = list(var_13)
    var_15 = None
    var_16 = module_0.ValidationResult(value=var_15, error=var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None

def test_case_39():
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
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_1, error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'typesystem.base.Message'
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = None
    var_7 = var_1.__hash__()
    assert var_7 == 5808716241442466140
    var_8 = [var_5, var_5, var_2, var_2]
    var_9 = None
    var_10 = var_1.__eq__(var_3)
    assert var_10 is False
    var_11 = None
    var_12 = module_0.ParseError(text=var_5, key=var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_12.__eq__(var_6)
    assert var_13 is False
    var_14 = var_12.__str__()
    assert var_14 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_15 = var_4.__iter__()
    var_16 = module_0.Message(text=var_6, index=var_0, start_position=var_11, end_position=var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text is None
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = var_12.__eq__(var_6)
    assert var_17 is False
    var_18 = var_17.__eq__(var_0)
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_8, error=var_17)

def test_case_40():
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
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_1, error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'typesystem.base.Message'
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = None
    var_7 = module_0.Position(var_2, var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is True
    var_8 = var_1.__hash__()
    assert var_8 == 5808716241442466140
    var_9 = var_4.__iter__()
    var_10 = None
    var_11 = var_1.__eq__(var_3)
    assert var_11 is False
    var_12 = var_4.__bool__()
    assert var_12 is True
    var_13 = var_7.__eq__(var_3)
    assert var_13 is False
    var_14 = module_0.ParseError(text=var_5, key=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.__eq__(var_6)
    assert var_15 is False
    var_16 = var_14.__eq__(var_3)
    assert var_16 is False
    var_17 = module_0.Message(text=var_6, code=var_10, position=var_7)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text is None
    assert var_17.code == 'custom'
    assert var_17.index == []
    assert f'{type(var_17.start_position).__module__}.{type(var_17.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_17.end_position).__module__}.{type(var_17.end_position).__qualname__}' == 'typesystem.base.Position'
    var_18 = var_17.__eq__(var_6)
    assert var_18 is False
    var_19 = var_14.__repr__()
    assert var_19 == 'ParseError(text="ValidationResult(value=Message(text=None, code=\'custom\'))", code=\'custom\')'
    var_20 = var_14.__str__()
    assert var_20 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_21 = var_17.__eq__(var_1)
    assert var_21 is False
    var_22 = var_14.__contains__(var_0)
    assert var_22 is False
    var_23 = var_22.__repr__()
    assert var_23 == 'False'
    var_24 = module_0.ParseError(text=var_23)
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_24) == 1
    var_25 = module_0.ValidationResult()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_25.value is None
    assert var_25.error is None
    var_26 = var_4.__bool__()
    assert var_26 is True

def test_case_41():
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
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_1, error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'typesystem.base.Message'
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = None
    var_7 = module_0.Position(var_2, var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is None
    assert var_7.char_index is True
    var_8 = var_1.__hash__()
    assert var_8 == 5808716241442466140
    var_9 = var_1.__eq__(var_3)
    assert var_9 is False
    var_10 = var_7.__eq__(var_3)
    assert var_10 is False
    var_11 = module_0.ParseError(text=var_5, key=var_3)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_11.__eq__(var_6)
    assert var_12 is False
    var_13 = var_11.__eq__(var_3)
    assert var_13 is False
    var_14 = var_11.__eq__(var_3)
    assert var_14 is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, position=var_7, end_position=var_9)

@pytest.mark.xfail(strict=True)
def test_case_42():
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
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_1, error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'typesystem.base.Message'
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = module_0.ParseError(text=var_5, code=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = None
    var_8 = module_0.Position(var_2, var_7, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = var_1.__hash__()
    assert var_9 == 5808716241442466140
    var_10 = None
    var_11 = '\tFH&Q+zc08xgxdqc@'
    var_12 = var_1.__eq__(var_3)
    assert var_12 is False
    var_13 = None
    var_14 = var_8.__eq__(var_3)
    assert var_14 is False
    var_15 = module_0.ParseError(text=var_5, key=var_3)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_15.__eq__(var_7)
    assert var_16 is False
    var_17 = var_15.__eq__(var_3)
    assert var_17 is False
    var_18 = module_0.Message(text=var_7, code=var_10, position=var_8)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Message'
    assert var_18.text is None
    assert var_18.code == 'custom'
    assert var_18.index == []
    assert f'{type(var_18.start_position).__module__}.{type(var_18.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_18.end_position).__module__}.{type(var_18.end_position).__qualname__}' == 'typesystem.base.Position'
    var_19 = var_15.values()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_19) == 1
    var_20 = var_18.__repr__()
    assert var_20 == "Message(text=None, code='custom', position=Position(line_no=True, column_no=None, char_index=True))"
    var_21 = var_6.__str__()
    assert var_21 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_22 = var_6.__len__()
    assert var_22 == 1
    var_23 = module_0.Message(text=var_13, key=var_11)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Message'
    assert var_23.text is None
    assert var_23.code == 'custom'
    assert var_23.index == ['\tFH&Q+zc08xgxdqc@']
    assert var_23.start_position is None
    assert var_23.end_position is None
    var_24 = var_23.__eq__(var_1)
    assert var_24 is False
    var_25 = var_6.__contains__(var_10)
    assert var_25 is False
    var_26 = var_22.__repr__()
    assert var_26 == '1'
    module_0.ParseError(key=var_0)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = None
    var_3 = module_0.ValidationResult(value=var_1, error=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_3.value).__module__}.{type(var_3.value).__qualname__}' == 'typesystem.base.Message'
    assert var_3.error is None
    var_4 = var_3.__repr__()
    assert var_4 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_5 = module_0.ParseError(text=var_4, code=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = None
    var_7 = module_0.Position(var_5, var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_7.line_no).__module__}.{type(var_7.line_no).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7.line_no) == 1
    assert var_7.column_no is None
    assert f'{type(var_7.char_index).__module__}.{type(var_7.char_index).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7.char_index) == 1
    var_8 = var_1.__hash__()
    assert var_8 == 5808716241442466140
    var_9 = None
    var_10 = var_1.__eq__(var_2)
    assert var_10 is False
    var_11 = None
    var_12 = var_7.__eq__(var_2)
    assert var_12 is False
    var_13 = var_11.__eq__(var_6)
    assert var_13 is True
    var_14 = var_5.__eq__(var_2)
    assert var_14 is False
    var_15 = module_0.Message(text=var_6, code=var_9, position=var_7)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text is None
    assert var_15.code == 'custom'
    assert var_15.index == []
    assert f'{type(var_15.start_position).__module__}.{type(var_15.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_15.end_position).__module__}.{type(var_15.end_position).__qualname__}' == 'typesystem.base.Position'
    var_16 = var_5.values()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_16) == 1
    var_17 = var_15.__repr__()
    assert var_17 == "Message(text=None, code='custom', start_position=Position(line_no=ValidationResult(value=Message(text=None, code='custom')), column_no=None, char_index=ValidationResult(value=Message(text=None, code='custom'))), end_position=Position(line_no=ValidationResult(value=Message(text=None, code='custom')), column_no=None, char_index=ValidationResult(value=Message(text=None, code='custom'))))"
    var_18 = var_16.__eq__(var_6)
    var_19 = var_5.get(var_9)
    var_19.__iter__()

def test_case_44():
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
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_1, error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'typesystem.base.Message'
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = module_0.ParseError(text=var_5, code=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = None
    var_8 = module_0.Position(var_2, var_7, var_2)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Position'
    assert var_8.line_no is True
    assert var_8.column_no is None
    assert var_8.char_index is True
    var_9 = None
    var_10 = var_1.__hash__()
    assert var_10 == 5808716241442466140
    var_11 = None
    var_12 = '\tFH&Q+zc08xgxdqc@'
    var_13 = var_1.__eq__(var_3)
    assert var_13 is False
    var_14 = None
    var_15 = var_8.__eq__(var_3)
    assert var_15 is False
    var_16 = module_0.ParseError(text=var_12, code=var_9)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_6.__eq__(var_14)
    assert var_17 is False
    var_18 = var_16.__eq__(var_3)
    assert var_18 is False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_5, code=var_11, position=var_2, start_position=var_15, end_position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_45():
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
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_1, error=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_4.value).__module__}.{type(var_4.value).__qualname__}' == 'typesystem.base.Message'
    assert var_4.error is None
    var_5 = var_4.__repr__()
    assert var_5 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_6 = module_0.ParseError(text=var_5, code=var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_6) == 1
    var_7 = module_0.Position(var_2, var_6, var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert f'{type(var_7.column_no).__module__}.{type(var_7.column_no).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7.column_no) == 1
    assert var_7.char_index is True
    var_8 = None
    var_9 = None
    var_10 = var_1.__eq__(var_3)
    assert var_10 is False
    var_11 = None
    var_12 = None
    var_13 = var_7.__eq__(var_3)
    assert var_13 is False
    var_14 = module_0.ParseError(text=var_5, key=var_3)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.__eq__(var_11)
    assert var_15 is False
    var_16 = var_14.__eq__(var_3)
    assert var_16 is False
    var_17 = module_0.Message(text=var_11, code=var_9, position=var_7)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text is None
    assert var_17.code == 'custom'
    assert var_17.index == []
    assert f'{type(var_17.start_position).__module__}.{type(var_17.start_position).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_17.end_position).__module__}.{type(var_17.end_position).__qualname__}' == 'typesystem.base.Position'
    var_18 = var_14.values()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_18) == 1
    var_19 = var_17.__repr__()
    assert var_19 == "Message(text=None, code='custom', start_position=Position(line_no=True, column_no=ValidationResult(value=Message(text=None, code='custom')), char_index=True), end_position=Position(line_no=True, column_no=ValidationResult(value=Message(text=None, code='custom')), char_index=True))"
    var_20 = var_18.__eq__(var_12)
    var_21 = var_14.__repr__()
    assert var_21 == 'ParseError(text="ValidationResult(value=Message(text=None, code=\'custom\'))", code=\'custom\')'
    var_22 = var_14.__str__()
    assert var_22 == "ValidationResult(value=Message(text=None, code='custom'))"
    var_23 = var_18.__eq__(var_18)
    assert var_23 is True
    var_24 = var_18.__contains__(var_0)
    assert var_24 is False
    var_25 = var_18.__repr__()
    assert var_25 == 'ValuesView(ParseError(text="ValidationResult(value=Message(text=None, code=\'custom\'))", code=\'custom\'))'
    module_0.ParseError(messages=var_8)