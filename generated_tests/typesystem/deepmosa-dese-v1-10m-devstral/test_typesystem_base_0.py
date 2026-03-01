# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

def test_case_1():
    var_0 = 'Gq\x0c=8m]9~J,a@YF'
    var_1 = module_0.BaseError(text=var_0, code=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == 'Gq\x0c=8m]9~J,a@YF'

def test_case_2():
    var_0 = 'GU31s.s'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'GU31s.s'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = None
    var_4 = None
    var_5 = module_0.ParseError(text=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = module_0.Message(text=var_0, index=var_3, end_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'GU31s.s'
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_2.__eq__(var_1)
    assert var_7 is False
    var_8 = var_5.__str__()
    assert var_8 == 'GU31s.s'
    var_9 = var_6.__eq__(var_1)
    assert var_9 is False
    var_10 = module_0.ParseError(text=var_0, key=var_7, position=var_7, messages=var_4)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = var_5.__contains__(var_1)
    assert var_12 is False
    var_13 = 695
    var_14 = var_6.__eq__(var_1)
    assert var_14 is False
    var_15 = module_0.Position(var_4, var_12, var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Position'
    assert var_15.line_no is None
    assert var_15.column_no is False
    assert var_15.char_index == 695
    var_16 = var_6.__eq__(var_11)
    assert var_16 is False
    var_17 = var_15.__eq__(var_11)
    assert var_17 is False
    var_18 = var_5.messages()

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = 'B(`!o+n(IF~Xz/)k8'
    var_1 = None
    var_2 = module_0.Message(text=var_0, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'B(`!o+n(IF~Xz/)k8'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='B(`!o+n(IF~Xz/)k8', code='custom')"
    module_0.ParseError(text=var_1, code=var_1)

@pytest.mark.xfail(strict=True)
def test_case_4():
    module_0.ParseError()

def test_case_5():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 1
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'code1'
    assert var_4.index == [1]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Eror 2'
    var_6 = 'code2'
    var_7 = 2
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Eror 2'
    assert var_9.code == 'code2'
    assert var_9.index == [2]
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = repr(var_11)

def test_case_6():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None

def test_case_7():
    var_0 = "fm'O&g~s0\n2"
    var_1 = None
    var_2 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert var_2.error == "fm'O&g~s0\n2"
    var_3 = var_2.__repr__()
    assert var_3 == 'ValidationResult(error="fm\'O&g~s0\\n2")'
    var_4 = var_0.__iter__()
    var_5 = var_4.__repr__()
    var_6 = 'uCi[%sQCz7'
    var_7 = module_0.Message(text=var_6, index=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text == 'uCi[%sQCz7'
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = None
    var_9 = module_0.Message(text=var_8, start_position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is None
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_9.__hash__()
    assert var_10 == -8846692290013397843

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = var_0.__iter__()
    var_3 = None
    var_4 = module_0.Message(text=var_3, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text is None
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_0.__iter__()
    var_6 = var_0.__repr__()
    assert var_6 == 'ValidationResult(value=None)'
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_4.__repr__()
    assert var_8 == "Message(text=None, code='custom')"
    var_9 = None
    var_10 = var_0.__repr__()
    assert var_10 == 'ValidationResult(value=None)'
    var_11 = None
    var_12 = "p'y*`y"
    var_13 = module_0.ParseError(text=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.__eq__(var_11)
    assert var_14 is False
    var_15 = var_14.__eq__(var_9)
    var_15.__iter__()

def test_case_9():
    var_0 = False
    var_1 = False
    var_2 = None
    var_3 = module_0.Position(var_0, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is False
    assert var_3.column_no is False
    assert var_3.char_index is None

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = "#_R(4K`'/>ad\x0bv[\\J"
    var_1 = module_0.ParseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_1) == 1
    var_1.__getitem__(var_1)

def test_case_11():
    var_0 = None
    var_1 = "#_R(4K`'/>ad\x0bv[\\J"
    var_2 = module_0.ParseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()
    var_4 = var_3.__eq__(var_0)
    var_5 = var_2.__len__()
    assert var_5 == 1

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'HnePPWyN_ZR'
    var_3 = module_0.Message(text=var_2, key=var_0, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'HnePPWyN_ZR'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_7 = module_0.Message(text=var_5, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is False
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = None
    var_10 = None
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_3.__eq__(var_10)
    assert var_12 is False
    var_13 = module_0.ParseError(text=var_4, key=var_5, position=var_5, messages=var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14) == 1
    var_15 = var_14.__repr__()
    assert var_15 == 'ItemsView(ParseError([Message(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\', index=[False], position=False)]))'
    var_16 = var_13.__contains__(var_10)
    assert var_16 is False
    var_17 = var_11.__bool__()
    assert var_17 is True
    var_14.messages()

def test_case_13():
    var_0 = 'Error message'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1

def test_case_14():
    var_0 = '=X{>0=9N_G&'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = None
    var_1 = 'B(`!o+n(IF~Xz/)k8'
    var_2 = module_0.Message(text=var_1, end_position=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'B(`!o+n(IF~Xz/)k8'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='B(`!o+n(IF~Xz/)k8', code='custom')"
    var_4 = [var_2, var_2, var_2]
    module_0.ParseError(code=var_1, messages=var_4)

def test_case_16():
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

def test_case_17():
    var_0 = []
    var_1 = None
    var_2 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == []
    assert var_2.column_no is None
    assert var_2.char_index is None
    var_3 = var_2.__repr__()
    assert var_3 == 'Position(line_no=[], column_no=None, char_index=None)'

def test_case_18():
    var_0 = 'j@>04\\Nyj`GO,0$'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_0, position=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = var_2.keys()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_3) == 1
    var_4 = module_0.Message(text=var_3, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_4.text).__module__}.{type(var_4.text).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4.text) == 1
    assert var_4.code == 'custom'
    assert f'{type(var_4.index).__module__}.{type(var_4.index).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4.index) == 1
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.values()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.ValuesView'
    assert len(var_5) == 1
    var_6 = var_5.__repr__()
    assert var_6 == "ValuesView(ValidationError(text='j@>04\\\\Nyj`GO,0$', code='j@>04\\\\Nyj`GO,0$'))"
    var_7 = var_2.__iter__()
    var_8 = var_7.__str__()
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False

def test_case_19():
    var_0 = '@Sa~zc4KkQ^\npN'
    var_1 = module_0.BaseError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.__eq__(var_0)
    assert var_2 is False

def test_case_20():
    var_0 = []
    var_1 = None
    var_2 = module_0.ValidationResult(value=var_0, error=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value == []
    assert var_2.error is None
    var_3 = module_0.Position(var_0, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no == []
    assert var_3.column_no is None
    assert var_3.char_index is None
    var_4 = var_3.__repr__()
    assert var_4 == 'Position(line_no=[], column_no=None, char_index=None)'

def test_case_21():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_5 = var_2.__hash__()
    assert var_5 == -8846692290013397843
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_7 = True
    var_8 = var_2.__repr__()
    assert var_8 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_2.__repr__()
    assert var_10 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_11 = None
    var_12 = '|<JW~#emV}o'
    var_13 = module_0.Position(var_11, var_1, var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.Position'
    assert var_13.line_no is None
    assert var_13.column_no is None
    assert var_13.char_index == -8846692290013397843
    with pytest.raises(AssertionError):
        module_0.Message(text=var_12, code=var_7, index=var_11, position=var_4, start_position=var_13, end_position=var_1)

def test_case_22():
    var_0 = {}
    var_1 = True
    var_2 = None
    var_3 = module_0.Position(var_1, var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Position'
    assert var_3.line_no is True
    assert var_3.column_no is True
    assert var_3.char_index is None
    var_4 = var_3.__eq__(var_0)
    assert var_4 is False

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_5 = var_2.__hash__()
    assert var_5 == -8846692290013397843
    var_6 = True
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_8 = None
    var_9 = module_0.Message(text=var_6, index=var_8, position=var_8, start_position=var_8, end_position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is True
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_9.__eq__(var_9)
    assert var_10 is True
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = module_0.ParseError(text=var_3)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_9.__repr__()
    assert var_13 == "Message(text=True, code='custom')"
    var_14 = module_0.Message(text=var_0, index=var_8, end_position=var_1)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'HnePPWyN_ZR'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = var_11.__repr__()
    assert var_15 == 'ValidationResult(value=None)'
    var_16 = True
    var_17 = var_14.__repr__()
    assert var_17 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_18 = var_12.__eq__(var_8)
    assert var_18 is False
    var_19 = var_12.__contains__(var_18)
    assert var_19 is False
    var_20 = var_19.__eq__(var_1)
    var_21 = var_12.messages(add_prefix=var_16)
    var_18.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = module_0.ValidationResult()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None
    var_2 = 'nn8*e+GaUb:0V3TW'
    var_3 = module_0.BaseError(text=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_3) == 1
    var_4 = var_3.messages(add_prefix=var_0)
    var_5 = var_3.__str__()
    assert var_5 == 'nn8*e+GaUb:0V3TW'
    module_0.ValidationError()

def test_case_25():
    var_0 = 'Gq\x0c=8m]9~J,a@YF'
    var_1 = module_0.BaseError(text=var_0, code=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_1) == 1
    var_2 = var_1.keys()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_2) == 1
    var_3 = var_2.__repr__()
    assert var_3 == "KeysView(BaseError(text='Gq\\x0c=8m]9~J,a@YF', code='Gq\\x0c=8m]9~J,a@YF'))"

def test_case_26():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_4 = True
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_6 = None
    var_7 = module_0.Message(text=var_4, index=var_6, position=var_6, start_position=var_6, end_position=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is True
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = module_0.ParseError(text=var_3)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.keys()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_11) == 1
    var_12 = var_10.get(var_6)
    var_13 = var_10.__eq__(var_11)
    assert var_13 is False
    var_14 = var_10.keys()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_14) == 1
    var_15 = var_11.__repr__()
    assert var_15 == 'KeysView(ParseError(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\'))'
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_13, error=var_14)

def test_case_27():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_4 = var_2.__hash__()
    assert var_4 == -8846692290013397843
    var_5 = True
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_7 = None
    var_8 = module_0.Message(text=var_5, index=var_7, position=var_7, start_position=var_7, end_position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is True
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_8)
    assert var_9 is True
    var_10 = []
    with pytest.raises(AssertionError):
        module_0.Message(text=var_3, key=var_0, index=var_10, start_position=var_7)

def test_case_28():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_5 = var_2.__hash__()
    assert var_5 == -8846692290013397843
    var_6 = True
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_8 = None
    var_9 = 'm_\x0bf;Tv\x0b'
    var_10 = module_0.Message(text=var_9, end_position=var_8)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'm_\x0bf;Tv\x0b'
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_10.__eq__(var_2)
    assert var_11 is False
    var_12 = None
    var_13 = module_0.ValidationResult(value=var_1, error=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = module_0.ParseError(text=var_9)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.__eq__(var_8)
    assert var_15 is False
    var_16 = var_14.keys()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_16) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_6, position=var_16, end_position=var_16)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__repr__()
    assert var_3 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_4 = var_2.__hash__()
    assert var_4 == -8846692290013397843
    var_5 = var_2.__repr__()
    assert var_5 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_6 = None
    var_7 = None
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_2.__eq__(var_1)
    assert var_9 is False
    var_10 = module_0.ParseError(text=var_3, key=var_4, position=var_4, messages=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_10) == 1
    var_11 = var_10.items()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_11) == 1
    var_12 = var_11.__repr__()
    assert var_12 == 'ItemsView(ParseError([Message(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\', index=[-8846692290013397843], position=-8846692290013397843)]))'
    var_13 = var_11.__repr__()
    assert var_13 == 'ItemsView(ParseError([Message(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\', index=[-8846692290013397843], position=-8846692290013397843)]))'
    var_11.__contains__(var_7)

def test_case_30():
    var_0 = 'GU31s.s'
    var_1 = None
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'GU31s.s'
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__hash__()
    assert var_3 == -8846692290013397843
    var_4 = None
    var_5 = None
    var_6 = module_0.ValidationResult()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert var_6.error is None
    var_7 = module_0.ParseError(text=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_7) == 1
    var_8 = None
    var_9 = module_0.Message(text=var_0, index=var_4, end_position=var_1)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'GU31s.s'
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_2.__eq__(var_1)
    assert var_10 is False
    var_11 = var_7.__str__()
    assert var_11 == 'GU31s.s'
    var_12 = module_0.ParseError(text=var_0, key=var_3, position=var_3, messages=var_5)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_12.__eq__(var_1)
    assert var_13 is False
    var_14 = var_7.__contains__(var_1)
    assert var_14 is False
    var_15 = 695
    var_16 = module_0.Position(var_5, var_14, var_15)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Position'
    assert var_16.line_no is None
    assert var_16.column_no is False
    assert var_16.char_index == 695
    var_17 = var_9.__eq__(var_13)
    assert var_17 is False
    var_18 = var_16.__eq__(var_13)
    assert var_18 is False
    var_19 = var_7.messages()
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_5, code=var_8, position=var_14, messages=var_13)

def test_case_31():
    var_0 = None
    var_1 = 'HnePPWyN_ZR'
    var_2 = None
    var_3 = module_0.Message(text=var_1, key=var_2, end_position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'HnePPWyN_ZR'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_5 = var_3.__eq__(var_3)
    assert var_5 is True
    var_6 = True
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_8 = None
    var_9 = module_0.Message(text=var_6, index=var_8, position=var_8, start_position=var_8, end_position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is True
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = var_9.__eq__(var_9)
    assert var_10 is True
    var_11 = None
    var_12 = None
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = module_0.ParseError(text=var_4)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_14) == 1
    var_15 = var_14.__eq__(var_11)
    assert var_15 is False
    var_16 = module_0.Message(text=var_1, index=var_8, end_position=var_2)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.Message'
    assert var_16.text == 'HnePPWyN_ZR'
    assert var_16.code == 'custom'
    assert var_16.index == []
    assert var_16.start_position is None
    assert var_16.end_position is None
    var_17 = var_3.__eq__(var_2)
    assert var_17 is False
    var_18 = var_14.__str__()
    assert var_18 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_19 = module_0.ParseError(text=var_4, key=var_5, position=var_5, messages=var_11)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_19) == 1
    var_20 = var_14.__str__()
    assert var_20 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_21 = var_19.items()
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_21) == 1
    var_22 = var_21.__repr__()
    assert var_22 == 'ItemsView(ParseError([Message(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\', index=[True], position=True)]))'
    var_23 = var_19.__eq__(var_2)
    assert var_23 is False
    var_24 = var_14.__contains__(var_12)
    assert var_24 is False
    var_25 = 691
    var_26 = module_0.Position(var_11, var_20, var_25)
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'typesystem.base.Position'
    assert var_26.line_no is None
    assert var_26.column_no == "Message(text='HnePPWyN_ZR', code='custom')"
    assert var_26.char_index == 691
    var_27 = var_19.__str__()
    assert var_27 == '{True: "Message(text=\'HnePPWyN_ZR\', code=\'custom\')"}'
    var_28 = var_14.messages()
    var_29 = module_0.BaseError(position=var_0, messages=var_28)
    assert f'{type(var_29).__module__}.{type(var_29).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_29) == 1

def test_case_32():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = []
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = 'lEQhO"97\r M-#='
    var_5 = None
    var_6 = module_0.Message(text=var_4, key=var_5, end_position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'lEQhO"97\r M-#='
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__repr__()
    assert var_7 == 'Message(text=\'lEQhO"97\\r M-#=\', code=\'custom\')'
    var_8 = var_6.__hash__()
    assert var_8 == -8846692290013397843
    var_9 = True
    var_10 = var_6.__repr__()
    assert var_10 == 'Message(text=\'lEQhO"97\\r M-#=\', code=\'custom\')'
    var_11 = None
    var_12 = module_0.Message(text=var_9, index=var_11, position=var_11, start_position=var_11, end_position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.Message'
    assert var_12.text is True
    assert var_12.code == 'custom'
    assert var_12.index == []
    assert var_12.start_position is None
    assert var_12.end_position is None
    var_13 = var_12.__eq__(var_12)
    assert var_13 is True
    var_14 = module_0.ValidationResult(error=var_5)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = None
    var_16 = None
    var_17 = module_0.ValidationResult()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert var_17.error is None
    var_18 = module_0.ParseError(text=var_7)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_18) == 1
    var_19 = var_18.__contains__(var_16)
    assert var_19 is False
    var_20 = module_0.Message(text=var_4, index=var_11, end_position=var_5)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Message'
    assert var_20.text == 'lEQhO"97\r M-#='
    assert var_20.code == 'custom'
    assert var_20.index == []
    assert var_20.start_position is None
    assert var_20.end_position is None
    var_21 = var_6.__eq__(var_5)
    assert var_21 is False
    var_22 = var_18.__str__()
    assert var_22 == 'Message(text=\'lEQhO"97\\r M-#=\', code=\'custom\')'
    var_23 = module_0.ParseError(text=var_7, key=var_8, position=var_8, messages=var_15)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_23) == 1
    var_24 = var_18.__repr__()
    assert var_24 == 'ParseError(text=\'Message(text=\\\'lEQhO"97\\\\r M-#=\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_25 = var_23.items()
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_25) == 1
    var_26 = var_25.__repr__()
    assert var_26 == 'ItemsView(ParseError([Message(text=\'Message(text=\\\'lEQhO"97\\\\r M-#=\\\', code=\\\'custom\\\')\', code=\'custom\', index=[-8846692290013397843], position=-8846692290013397843)]))'
    var_27 = var_23.__eq__(var_5)
    assert var_27 is False
    var_28 = var_18.__contains__(var_16)
    assert var_28 is False
    var_29 = 695
    var_30 = module_0.Position(var_15, var_24, var_29)
    assert f'{type(var_30).__module__}.{type(var_30).__qualname__}' == 'typesystem.base.Position'
    assert var_30.line_no is None
    assert var_30.column_no == 'ParseError(text=\'Message(text=\\\'lEQhO"97\\\\r M-#=\\\', code=\\\'custom\\\')\', code=\'custom\')'
    assert var_30.char_index == 695
    var_31 = var_20.__eq__(var_27)
    assert var_31 is False
    var_32 = var_30.__eq__(var_27)
    assert var_32 is False
    with pytest.raises(AssertionError):
        module_0.BaseError(position=var_0, messages=var_2)

def test_case_33():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'HnePPWyN_ZR'
    var_3 = None
    var_4 = module_0.Message(text=var_2, key=var_3, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'HnePPWyN_ZR'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_6 = var_4.__eq__(var_1)
    assert var_6 is False
    var_7 = True
    var_8 = var_4.__repr__()
    assert var_8 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_9 = None
    var_10 = module_0.Message(text=var_7, index=var_9, position=var_9, start_position=var_9, end_position=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is True
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_10.__eq__(var_10)
    assert var_11 is True
    var_12 = None
    var_13 = None
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = module_0.ParseError(text=var_5)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_15.__eq__(var_12)
    assert var_16 is False
    var_17 = module_0.Message(text=var_2, index=var_9, end_position=var_3)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.Message'
    assert var_17.text == 'HnePPWyN_ZR'
    assert var_17.code == 'custom'
    assert var_17.index == []
    assert var_17.start_position is None
    assert var_17.end_position is None
    var_18 = var_4.__eq__(var_3)
    assert var_18 is False
    var_19 = var_15.__str__()
    assert var_19 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_20 = module_0.ParseError(text=var_5, key=var_6, position=var_6, messages=var_12)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_20) == 1
    var_21 = var_15.__repr__()
    assert var_21 == 'ParseError(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\')'
    var_22 = var_20.items()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_22) == 1
    var_23 = var_22.__repr__()
    assert var_23 == 'ItemsView(ParseError([Message(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\', index=[False], position=False)]))'
    var_24 = var_20.__eq__(var_3)
    assert var_24 is False
    var_25 = var_15.__contains__(var_13)
    assert var_25 is False
    var_26 = 695
    var_27 = module_0.Position(var_12, var_21, var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.base.Position'
    assert var_27.line_no is None
    assert var_27.column_no == 'ParseError(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\')'
    assert var_27.char_index == 695
    var_28 = var_17.__eq__(var_24)
    assert var_28 is False
    var_29 = var_20.__str__()
    assert var_29 == '{False: "Message(text=\'HnePPWyN_ZR\', code=\'custom\')"}'
    var_30 = var_27.__eq__(var_24)
    assert var_30 is False
    var_31 = var_15.messages()
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_23, messages=var_22)

def test_case_34():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'HnePPWyN_ZR'
    var_3 = None
    var_4 = module_0.Message(text=var_2, key=var_3, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'HnePPWyN_ZR'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_4.__repr__()
    assert var_5 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_6 = True
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_8 = module_0.Message(text=var_6, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is True
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_8)
    assert var_9 is True
    var_10 = None
    var_11 = None
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = module_0.ParseError(text=var_5)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_13.__iter__()
    var_15 = var_4.__eq__(var_3)
    assert var_15 is False
    var_16 = module_0.ParseError(text=var_5, key=var_13, position=var_13, messages=var_10)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_16.items()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_17) == 1
    var_18 = var_17.__repr__()
    assert var_18 == 'ItemsView(ParseError([Message(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\', index=[ParseError(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\')], start_position=ParseError(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\'), end_position=ParseError(text="Message(text=\'HnePPWyN_ZR\', code=\'custom\')", code=\'custom\'))]))'
    var_19 = var_16.__eq__(var_3)
    assert var_19 is False
    var_20 = var_13.__contains__(var_11)
    assert var_20 is False
    var_21 = 695
    var_22 = module_0.Position(var_10, var_5, var_21)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.Position'
    assert var_22.line_no is None
    assert var_22.column_no == "Message(text='HnePPWyN_ZR', code='custom')"
    assert var_22.char_index == 695
    var_23 = var_12.__bool__()
    assert var_23 is True
    var_24 = var_22.__eq__(var_19)
    assert var_24 is False
    var_25 = var_13.messages()

def test_case_35():
    var_0 = None
    var_1 = 538
    var_2 = module_0.Position(var_1, var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Position'
    assert var_2.line_no == 538
    assert var_2.column_no == 538
    assert var_2.char_index == 538
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True
    var_4 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Position'
    assert var_4.line_no is None
    assert var_4.column_no is None
    assert var_4.char_index is None
    var_5 = True
    var_6 = True
    var_7 = module_0.Position(var_5, var_6, var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is True
    assert var_7.column_no is True
    assert var_7.char_index is None

def test_case_36():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = []
    var_3 = var_1.__eq__(var_2)
    assert var_3 is False
    var_4 = 'lEQhO"97\r M-#='
    var_5 = None
    var_6 = module_0.Message(text=var_4, key=var_5, end_position=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'lEQhO"97\r M-#='
    assert var_6.code == 'custom'
    assert var_6.index == []
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = var_6.__repr__()
    assert var_7 == 'Message(text=\'lEQhO"97\\r M-#=\', code=\'custom\')'
    var_8 = var_6.__hash__()
    assert var_8 == -8846692290013397843
    var_9 = var_6.__repr__()
    assert var_9 == 'Message(text=\'lEQhO"97\\r M-#=\', code=\'custom\')'
    var_10 = None
    var_11 = module_0.Message(text=var_3, index=var_10, position=var_10, start_position=var_10, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is False
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_11.__eq__(var_11)
    assert var_12 is True
    var_13 = None
    var_14 = None
    var_15 = module_0.ValidationResult()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is None
    var_16 = module_0.ParseError(text=var_7)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_16) == 1
    var_17 = var_16.__contains__(var_14)
    assert var_17 is False
    var_18 = var_17.__repr__()
    assert var_18 == 'False'
    var_19 = module_0.Message(text=var_4, index=var_10, end_position=var_5)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.base.Message'
    assert var_19.text == 'lEQhO"97\r M-#='
    assert var_19.code == 'custom'
    assert var_19.index == []
    assert var_19.start_position is None
    assert var_19.end_position is None
    var_20 = var_6.__eq__(var_5)
    assert var_20 is False
    var_21 = var_16.__str__()
    assert var_21 == 'Message(text=\'lEQhO"97\\r M-#=\', code=\'custom\')'
    var_22 = module_0.ParseError(text=var_7, key=var_8, position=var_8, messages=var_13)
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_22) == 1
    var_23 = var_16.__repr__()
    assert var_23 == 'ParseError(text=\'Message(text=\\\'lEQhO"97\\\\r M-#=\\\', code=\\\'custom\\\')\', code=\'custom\')'
    var_24 = var_22.items()
    assert f'{type(var_24).__module__}.{type(var_24).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_24) == 1
    var_25 = var_24.__repr__()
    assert var_25 == 'ItemsView(ParseError([Message(text=\'Message(text=\\\'lEQhO"97\\\\r M-#=\\\', code=\\\'custom\\\')\', code=\'custom\', index=[-8846692290013397843], position=-8846692290013397843)]))'
    var_26 = var_22.__eq__(var_5)
    assert var_26 is False
    var_27 = 687
    var_28 = module_0.Position(var_13, var_23, var_27)
    assert f'{type(var_28).__module__}.{type(var_28).__qualname__}' == 'typesystem.base.Position'
    assert var_28.line_no is None
    assert var_28.column_no == 'ParseError(text=\'Message(text=\\\'lEQhO"97\\\\r M-#=\\\', code=\\\'custom\\\')\', code=\'custom\')'
    assert var_28.char_index == 687
    var_29 = var_28.__eq__(var_26)
    assert var_29 is False
    var_30 = var_24.__contains__(var_15)
    assert var_30 is False
    var_31 = var_16.messages()
    var_32 = module_0.BaseError(position=var_0, messages=var_31)
    assert f'{type(var_32).__module__}.{type(var_32).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_32) == 1

def test_case_37():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'HnePyN_ZR'
    var_3 = module_0.Message(text=var_2, key=var_0, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'HnePyN_ZR'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='HnePyN_ZR', code='custom')"
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False
    var_6 = True
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text='HnePyN_ZR', code='custom')"
    var_8 = module_0.Message(text=var_6, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is True
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_8.__eq__(var_8)
    assert var_9 is True
    var_10 = None
    var_11 = None
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = module_0.ParseError(text=var_4)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = module_0.Message(text=var_2, index=var_10, end_position=var_10)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.Message'
    assert var_14.text == 'HnePyN_ZR'
    assert var_14.code == 'custom'
    assert var_14.index == []
    assert var_14.start_position is None
    assert var_14.end_position is None
    var_15 = var_3.__eq__(var_0)
    assert var_15 is False
    var_16 = var_13.__str__()
    assert var_16 == "Message(text='HnePyN_ZR', code='custom')"
    var_17 = module_0.ParseError(text=var_4, key=var_5, position=var_5, messages=var_10)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_17) == 1
    var_18 = var_13.__repr__()
    assert var_18 == 'ParseError(text="Message(text=\'HnePyN_ZR\', code=\'custom\')", code=\'custom\')'
    var_19 = var_17.keys()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_19) == 1
    var_20 = module_0.Position(var_0, var_11, var_15)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert var_20.line_no is None
    assert var_20.column_no is None
    assert var_20.char_index is False
    var_21 = var_13.__contains__(var_11)
    assert var_21 is False
    var_22 = 695
    var_23 = module_0.Position(var_10, var_18, var_22)
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.base.Position'
    assert var_23.line_no is None
    assert var_23.column_no == 'ParseError(text="Message(text=\'HnePyN_ZR\', code=\'custom\')", code=\'custom\')'
    assert var_23.char_index == 695
    var_24 = var_23.__eq__(var_20)
    assert var_24 is False
    var_25 = var_13.messages()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = None
    var_1 = module_0.Message(text=var_0, code=var_0, key=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = 'HnePyN_ZR'
    var_3 = module_0.Message(text=var_2, key=var_0, end_position=var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'HnePyN_ZR'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_3.__repr__()
    assert var_4 == "Message(text='HnePyN_ZR', code='custom')"
    var_5 = var_3.__eq__(var_1)
    assert var_5 is False
    var_6 = var_3.__repr__()
    assert var_6 == "Message(text='HnePyN_ZR', code='custom')"
    var_7 = module_0.Message(text=var_5, index=var_0, position=var_0, start_position=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is False
    assert var_7.code == 'custom'
    assert var_7.index == []
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_7.__eq__(var_7)
    assert var_8 is True
    var_9 = None
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = module_0.ParseError(text=var_4)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_11.__iter__()
    var_13 = var_3.__eq__(var_0)
    assert var_13 is False
    var_14 = var_11.__str__()
    assert var_14 == "Message(text='HnePyN_ZR', code='custom')"
    var_15 = module_0.ParseError(text=var_4, key=var_5, position=var_5, messages=var_9)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_11.__repr__()
    assert var_16 == 'ParseError(text="Message(text=\'HnePyN_ZR\', code=\'custom\')", code=\'custom\')'
    var_17 = var_15.keys()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_17) == 1
    var_18 = module_0.Position(var_0, var_9, var_13)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.Position'
    assert var_18.line_no is None
    assert var_18.column_no is None
    assert var_18.char_index is False
    var_19 = var_11.__contains__(var_9)
    assert var_19 is False
    var_20 = module_0.Position(var_12, var_16, var_12)
    assert f'{type(var_20).__module__}.{type(var_20).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_20.line_no).__module__}.{type(var_20.line_no).__qualname__}' == 'builtins.dict_keyiterator'
    assert var_20.column_no == 'ParseError(text="Message(text=\'HnePyN_ZR\', code=\'custom\')", code=\'custom\')'
    assert f'{type(var_20.char_index).__module__}.{type(var_20.char_index).__qualname__}' == 'builtins.dict_keyiterator'
    var_21 = var_10.__bool__()
    assert var_21 is True
    var_22 = var_20.__eq__(var_18)
    assert var_22 is False
    var_23 = var_10.__repr__()
    assert var_23 == 'ValidationResult(value=None)'
    var_24 = [var_7, var_19, var_7, var_18]
    module_0.BaseError(position=var_0, messages=var_24)

def test_case_39():
    var_0 = 'First error'
    var_1 = 'error1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'First error'
    assert var_4.code == 'error1'
    assert var_4.index == [0]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Second error'
    var_6 = 'error2'
    var_7 = 1
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Second error'
    assert var_9.code == 'error2'
    assert var_9.index == [1]
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = str(var_11)
    assert var_12 == "{0: 'First error', 1: 'Second error'}"

def test_case_40():
    var_0 = 'First error'
    var_1 = 'error1'
    var_2 = 0
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'First error'
    assert var_4.code == 'error1'
    assert var_4.index == [0]
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Second error'
    var_6 = 'error2'
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text='First error', code='error1', index=[0])"
    var_8 = 1
    var_9 = [var_8, var_8]
    var_10 = module_0.Message(text=var_5, code=var_6, index=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text == 'Second error'
    assert var_10.code == 'error2'
    assert var_10.index == [1, 1]
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = [var_4, var_10]
    var_12 = module_0.BaseError(messages=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_12) == 2
    var_13 = str(var_12)

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = None
    var_1 = module_0.Message(text=var_0, key=var_0, position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None
    var_2 = var_1.__repr__()
    assert var_2 == "Message(text=None, code='custom')"
    var_3 = var_1.__hash__()
    assert var_3 == -8846692290013397843
    var_4 = True
    var_5 = var_1.__repr__()
    assert var_5 == "Message(text=None, code='custom')"
    var_6 = None
    var_7 = module_0.Message(text=var_6, key=var_4, index=var_0, end_position=var_0)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert var_7.text is None
    assert var_7.code == 'custom'
    assert var_7.index == [True]
    assert var_7.start_position is None
    assert var_7.end_position is None
    var_8 = var_1.__eq__(var_7)
    assert var_8 is False
    var_9 = module_0.ValidationResult(error=var_0)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    module_0.ParseError()