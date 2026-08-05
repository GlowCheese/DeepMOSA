# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.base as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Message(text=var_0, end_position=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Message'
    assert var_1.text is None
    assert var_1.code == 'custom'
    assert var_1.index == []
    assert var_1.start_position is None
    assert var_1.end_position is None

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = 'j@>04\\Nyj`GO,0$'
    var_2 = None
    var_3 = module_0.ValidationError(text=var_1, code=var_1, position=var_2, messages=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3) == 1
    var_4 = var_3.get(var_0)
    var_4.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = 'HePWyN_ZR+\\'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HePWyN_ZR+\\'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HePWyN_ZR+\\'
    var_4 = var_2.__eq__(var_2)
    assert var_4 is True
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text='HePWyN_ZR+\\\\', code='custom', index=[ValidationError(text='HePWyN_ZR+\\\\', code='custom')])"
    var_7 = var_6.__hash__()
    assert var_7 == -3366403694558235138
    module_0.ValidationError()

def test_case_3():
    var_0 = None
    var_1 = None
    var_2 = 'B(`!o+n(IF~Xz/)k8'
    var_3 = None
    var_4 = module_0.Message(text=var_2, end_position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'B(`!o+n(IF~Xz/)k8'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = module_0.Message(text=var_1, index=var_1, end_position=var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_4.__repr__()
    assert var_6 == "Message(text='B(`!o+n(IF~Xz/)k8', code='custom')"
    var_7 = var_4.__repr__()
    assert var_7 == "Message(text='B(`!o+n(IF~Xz/)k8', code='custom')"

def test_case_4():
    var_0 = 'ln2ie1col1'
    var_1 = 'line1:col5'
    var_2 = 'error'
    var_3 = module_0.Message(text=var_2, start_position=var_0, end_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'error'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position == 'ln2ie1col1'
    assert var_3.end_position == 'line1:col5'
    var_4 = repr(var_3)

def test_case_5():
    var_0 = None
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, code=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
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
    var_5 = module_0.ParseError(code=var_0, position=var_0, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_3.keys()

def test_case_7():
    var_0 = None
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value is None
    assert var_1.error is None

def test_case_8():
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
    assert var_10 == 6097441371710409680

def test_case_9():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success'
    assert var_1.error is None
    var_2 = list(var_1)
    var_3 = var_1.__repr__()
    assert var_3 == "ValidationResult(value='success')"
    var_4 = len(var_2)
    assert var_4 == 2

def test_case_10():
    var_0 = None
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no is None
    assert var_1.column_no is None
    assert var_1.char_index is None

def test_case_11():
    var_0 = None
    var_1 = "#_R(4K`'/>ad\x0bv[\\J"
    var_2 = module_0.ParseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()
    var_4 = var_3.__eq__(var_0)

def test_case_12():
    var_0 = module_0.ValidationResult()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_0.value is None
    assert var_0.error is None
    var_1 = var_0.__bool__()
    assert var_1 is True
    var_2 = var_0.__bool__()
    assert var_2 is True
    var_3 = var_0.__iter__()
    var_4 = None
    var_5 = module_0.Message(text=var_4, index=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Message'
    assert var_5.text is None
    assert var_5.code == 'custom'
    assert var_5.index == []
    assert var_5.start_position is None
    assert var_5.end_position is None
    var_6 = var_0.__iter__()
    var_7 = var_0.__repr__()
    assert var_7 == 'ValidationResult(value=None)'
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    with pytest.raises(AssertionError):
        module_0.BaseError(key=var_1)

def test_case_13():
    var_0 = 'Xv'
    var_1 = None
    var_2 = module_0.ParseError(text=var_0, key=var_1, position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.ValidationResult(value=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert var_4.error is None

def test_case_14():
    var_0 = '=X{>0=9N_G&'
    with pytest.raises(AssertionError):
        module_0.BaseError(text=var_0, messages=var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_4 = [var_2, var_2, var_2]
    module_0.ParseError(code=var_0, messages=var_4)

def test_case_16():
    var_0 = False
    var_1 = None
    var_2 = module_0.Message(text=var_0, index=var_1, position=var_1, start_position=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text is False
    assert var_2.code == 'custom'
    assert var_2.index == []
    assert var_2.start_position is None
    assert var_2.end_position is None
    var_3 = var_2.__eq__(var_2)
    assert var_3 is True

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
    var_0 = None
    var_1 = "#_R(4K`'/>ad\x0bv[\\J"
    var_2 = module_0.ParseError(text=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_2) == 1
    var_3 = var_2.__iter__()
    var_4 = var_3.__eq__(var_0)
    var_5 = var_2.__len__()
    assert var_5 == 1

def test_case_19():
    var_0 = 'Error 1'
    var_1 = 'c1'
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = module_0.Message(text=var_0, code=var_1, index=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'Error 1'
    assert var_4.code == 'c1'
    assert var_4.index == ['field1']
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = 'Error 2'
    var_6 = 'c2'
    var_7 = 'field2'
    var_8 = [var_7]
    var_9 = module_0.Message(text=var_5, code=var_6, index=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'c2'
    assert var_9.index == ['field2']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_4, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 2
    var_12 = var_11.__str__()
    assert var_12 == "{'field1': 'Error 1', 'field2': 'Error 2'}"

def test_case_20():
    var_0 = 'success'
    var_1 = module_0.ValidationResult(value=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_1.value == 'success'
    assert var_1.error is None
    var_2 = list(var_1)

def test_case_21():
    var_0 = False
    with pytest.raises(AssertionError):
        module_0.Message(text=var_0, key=var_0, index=var_0, end_position=var_0)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_5 = var_1.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5) == 1
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_7 = None
    var_8 = module_0.Message(text=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is None
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_3.__hash__()
    assert var_9 == 1
    var_10 = var_8.__eq__(var_7)
    assert var_10 is False
    module_0.ParseError(messages=var_7)

def test_case_23():
    var_0 = 'Error 1'
    var_1 = 'code1'
    var_2 = 'users'
    var_3 = 0
    var_4 = 'name'
    var_5 = [var_2, var_3, var_4]
    var_6 = module_0.Message(text=var_0, code=var_1, index=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'Error 1'
    assert var_6.code == 'code1'
    assert var_6.index == ['users', 0, 'name']
    assert var_6.start_position is None
    assert var_6.end_position is None
    var_7 = 'Error 2'
    var_8 = 'code2'
    var_9 = module_0.Message(text=var_7, code=var_8, index=var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text == 'Error 2'
    assert var_9.code == 'code2'
    assert var_9.index == ['users', 0, 'name']
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = [var_6, var_9]
    var_11 = module_0.BaseError(messages=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.BaseError'
    assert len(var_11) == 1
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_11.messages()
    var_14 = len(var_13)
    assert var_14 == 2

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'HnePPWyN_ZR'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.__repr__()
    assert var_4 == "ValidationError(text='HnePPWyN_ZR', code='custom')"
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_6 = var_2.__eq__(var_1)
    assert var_6 is False
    var_7 = True
    var_8 = var_3.__repr__()
    assert var_8 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_9 = None
    var_10 = module_0.Message(text=var_7, index=var_9, position=var_9, start_position=var_9, end_position=var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Message'
    assert var_10.text is True
    assert var_10.code == 'custom'
    assert var_10.index == []
    assert var_10.start_position is None
    assert var_10.end_position is None
    var_11 = var_3.__hash__()
    assert var_11 == 6097441371710409680
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_5 = var_1.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5) == 1
    var_6 = True
    var_7 = None
    var_8 = module_0.Message(text=var_6, index=var_7, position=var_7, start_position=var_7, end_position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is True
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_2.__hash__()
    assert var_9 == -6602527800710040010
    var_10 = var_8.__eq__(var_8)
    assert var_10 is True
    var_11 = None
    var_12 = var_2.__repr__()
    assert var_12 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_13 = var_1.__iter__()
    var_14 = var_5.__eq__(var_13)
    var_9.get(var_3, var_11)

def test_case_26():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text == 'HnePPWyN_ZR'
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert var_3.start_position is None
    assert var_3.end_position is None
    var_4 = var_2.__len__()
    assert var_4 == 1
    var_5 = var_3.__repr__()
    assert var_5 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_6 = var_3.__hash__()
    assert var_6 == 6097441371710409680
    var_7 = var_3.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom')"
    with pytest.raises(AssertionError):
        module_0.ValidationResult(value=var_5, error=var_4)

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = var_2.__repr__()
    assert var_6 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_7 = None
    var_8 = module_0.ParseError(text=var_6, code=var_4, position=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_8) == 1
    var_9 = var_1.items()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_9) == 1
    module_0.ValidationError(text=var_2, code=var_4, key=var_7, messages=var_2)

def test_case_28():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = var_4.__repr__()
    assert var_5 == "KeysView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    var_6 = var_4.__repr__()
    assert var_6 == "KeysView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    var_7 = module_0.Message(text=var_2, key=var_4, position=var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Message'
    assert f'{type(var_7.text).__module__}.{type(var_7.text).__qualname__}' == 'typesystem.base.Message'
    assert var_7.code == 'custom'
    assert f'{type(var_7.index).__module__}.{type(var_7.index).__qualname__}' == 'builtins.list'
    assert len(var_7.index) == 1
    assert f'{type(var_7.start_position).__module__}.{type(var_7.start_position).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7.start_position) == 1
    assert f'{type(var_7.end_position).__module__}.{type(var_7.end_position).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_7.end_position) == 1
    var_8 = var_2.__hash__()
    assert var_8 == -6602527800710040010
    var_9 = None
    var_10 = var_7.__eq__(var_9)
    assert var_10 is False
    var_11 = None
    var_12 = var_1.__str__()
    assert var_12 == 'HnePPWyN_ZR'
    var_13 = module_0.ParseError(text=var_6, position=var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_11, position=var_12, start_position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = var_1.__str__()
    assert var_2 == 'HnePPWyN_ZR'
    var_3 = None
    var_4 = None
    var_5 = module_0.ValidationResult()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert var_5.error is None
    module_0.ParseError(key=var_4, position=var_3)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_5 = var_1.messages(add_prefix=var_3)
    var_6 = True
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_8 = None
    var_9 = module_0.Message(text=var_6, index=var_8, position=var_8, start_position=var_8, end_position=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Message'
    assert var_9.text is True
    assert var_9.code == 'custom'
    assert var_9.index == []
    assert var_9.start_position is None
    assert var_9.end_position is None
    var_10 = module_0.ValidationResult()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert var_10.error is None
    var_11 = var_2.__hash__()
    assert var_11 == -6602527800710040010
    var_12 = var_9.__eq__(var_9)
    assert var_12 is True
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_5 = var_1.keys()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_5) == 1
    var_6 = True
    var_7 = None
    var_8 = module_0.Message(text=var_6, index=var_7, position=var_7, start_position=var_7, end_position=var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.Message'
    assert var_8.text is True
    assert var_8.code == 'custom'
    assert var_8.index == []
    assert var_8.start_position is None
    assert var_8.end_position is None
    var_9 = var_2.__hash__()
    assert var_9 == -6602527800710040010
    var_10 = var_8.__eq__(var_8)
    assert var_10 is True
    var_11 = var_1.__str__()
    assert var_11 == 'HnePPWyN_ZR'
    module_0.ParseError(key=var_3, position=var_5, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = None
    var_3 = var_1.messages(add_prefix=var_2)
    var_4 = None
    var_5 = var_1.__eq__(var_4)
    assert var_5 is False
    var_6 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Message'
    assert var_6.text == 'HnePPWyN_ZR'
    assert var_6.code == 'custom'
    assert f'{type(var_6.index).__module__}.{type(var_6.index).__qualname__}' == 'builtins.list'
    assert len(var_6.index) == 1
    assert var_6.start_position is None
    assert f'{type(var_6.end_position).__module__}.{type(var_6.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.end_position) == 1
    var_7 = var_1.__len__()
    assert var_7 == 1
    var_8 = var_6.__repr__()
    assert var_8 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_9 = var_1.keys()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_9) == 1
    var_10 = None
    var_11 = module_0.Message(text=var_4, index=var_10, position=var_10, start_position=var_10, end_position=var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.Message'
    assert var_11.text is None
    assert var_11.code == 'custom'
    assert var_11.index == []
    assert var_11.start_position is None
    assert var_11.end_position is None
    var_12 = var_6.__hash__()
    assert var_12 == -6602527800710040010
    var_13 = var_11.__eq__(var_11)
    assert var_13 is True
    var_14 = None
    var_15 = module_0.ValidationResult(value=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is None
    module_0.ParseError(key=var_10, position=var_7, messages=var_7)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = var_2.__eq__(var_2)
    assert var_5 is True
    var_6 = None
    var_7 = var_2.__repr__()
    assert var_7 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_8 = var_4.__iter__()
    var_9 = var_1.__iter__()
    var_10 = [var_4, var_8]
    var_11 = module_0.ParseError(text=var_7, code=var_10, position=var_8)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_1.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12) == 1
    var_13 = var_12.__repr__()
    assert var_13 == "ItemsView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    var_14 = []
    module_0.ParseError(position=var_6, messages=var_14)

@pytest.mark.xfail(strict=True)
def test_case_34():
    var_0 = 'E7ik) >T_*x'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'E7ik) >T_*x'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'E7ik) >T_*x'
    var_4 = var_2.__repr__()
    assert var_4 == "Message(text='E7ik) >T_*x', code='custom', index=[ValidationError(text='E7ik) >T_*x', code='custom')])"
    var_5 = None
    var_6 = var_2.__eq__(var_2)
    assert var_6 is True
    var_7 = module_0.ValidationResult()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert var_7.error is None
    var_8 = var_2.__repr__()
    assert var_8 == "Message(text='E7ik) >T_*x', code='custom', index=[ValidationError(text='E7ik) >T_*x', code='custom')])"
    var_9 = var_4.__iter__()
    var_10 = [var_4, var_9]
    var_11 = module_0.ParseError(text=var_8, code=var_10, position=var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_11) == 1
    var_12 = var_1.items()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_12) == 1
    var_13 = var_11.__iter__()
    var_14 = var_7.__repr__()
    assert var_14 == 'ValidationResult(value=None)'
    var_15 = module_0.Message(text=var_5, index=var_4, position=var_4)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.Message'
    assert var_15.text is None
    assert var_15.code == 'custom'
    assert var_15.index == "Message(text='E7ik) >T_*x', code='custom', index=[ValidationError(text='E7ik) >T_*x', code='custom')])"
    assert var_15.start_position == "Message(text='E7ik) >T_*x', code='custom', index=[ValidationError(text='E7ik) >T_*x', code='custom')])"
    assert var_15.end_position == "Message(text='E7ik) >T_*x', code='custom', index=[ValidationError(text='E7ik) >T_*x', code='custom')])"
    var_16 = var_15.__repr__()
    assert var_16 == 'Message(text=None, code=\'custom\', index="Message(text=\'E7ik) >T_*x\', code=\'custom\', index=[ValidationError(text=\'E7ik) >T_*x\', code=\'custom\')])", position="Message(text=\'E7ik) >T_*x\', code=\'custom\', index=[ValidationError(text=\'E7ik) >T_*x\', code=\'custom\')])")'
    module_0.ParseError(key=var_4, messages=var_5)

@pytest.mark.xfail(strict=True)
def test_case_35():
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
    var_5 = module_0.ParseError(code=var_0, position=var_0, messages=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_5) == 1
    var_6 = var_5.items()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_6) == 1
    var_7 = var_6.__repr__()
    assert var_7 == "ItemsView(ParseError([Message(text='B(`!o+n(IF~Xz/)k8', code='custom'), Message(text='B(`!o+n(IF~Xz/)k8', code='custom'), Message(text='B(`!o+n(IF~Xz/)k8', code='custom')]))"
    var_3.keys()

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = None
    var_5 = 609
    var_6 = module_0.Position(var_3, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert var_6.line_no == 'HnePPWyN_ZR'
    assert var_6.column_no == 609
    assert var_6.char_index == 609
    var_7 = var_6.__eq__(var_4)
    assert var_7 is False
    var_8 = var_2.__repr__()
    assert var_8 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_9 = var_2.__eq__(var_2)
    assert var_9 is True
    var_10 = None
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_2.__repr__()
    assert var_12 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_13 = module_0.ParseError(text=var_12, code=var_8, position=var_10)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_13) == 1
    var_14 = var_1.items()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_14) == 1
    var_15 = var_11.__repr__()
    assert var_15 == 'ValidationResult(value=None)'
    var_16 = var_14.__repr__()
    assert var_16 == "ItemsView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    module_0.ParseError(position=var_4, messages=var_14)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = 740
    var_6 = module_0.Position(var_4, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6.line_no) == 1
    assert var_6.column_no == 740
    assert var_6.char_index == 740
    var_7 = var_6.__eq__(var_4)
    assert var_7 is False
    var_8 = None
    var_9 = var_2.__eq__(var_2)
    assert var_9 is True
    var_10 = None
    var_11 = module_0.ValidationResult()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert var_11.error is None
    var_12 = var_2.__repr__()
    assert var_12 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_13 = var_4.__repr__()
    assert var_13 == "KeysView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    var_14 = var_4.__iter__()
    var_15 = module_0.ParseError(text=var_14, key=var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_15.items()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16) == 1
    var_17 = var_14.__repr__()
    var_18 = module_0.ValidationResult(value=var_16)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'typesystem.base.ValidationResult'
    assert f'{type(var_18.value).__module__}.{type(var_18.value).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_18.value) == 1
    assert var_18.error is None
    var_19 = var_18.__repr__()
    module_0.ParseError(key=var_10, position=var_8)

def test_case_38():
    var_0 = 743
    var_1 = module_0.Position(var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.Position'
    assert var_1.line_no == 743
    assert var_1.column_no == 743
    assert var_1.char_index == 743
    var_2 = var_1.__eq__(var_1)
    assert var_2 is True

def test_case_39():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = var_1.keys()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_4) == 1
    var_5 = 740
    var_6 = module_0.Position(var_4, var_5, var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.Position'
    assert f'{type(var_6.line_no).__module__}.{type(var_6.line_no).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_6.line_no) == 1
    assert var_6.column_no == 740
    assert var_6.char_index == 740
    var_7 = var_6.__eq__(var_4)
    assert var_7 is False
    var_8 = var_2.__eq__(var_2)
    assert var_8 is True
    var_9 = module_0.ValidationResult()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert var_9.error is None
    var_10 = var_2.__repr__()
    assert var_10 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_11 = None
    var_12 = var_4.__repr__()
    assert var_12 == "KeysView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    var_13 = var_4.__iter__()
    var_14 = [var_4, var_13]
    var_15 = module_0.ParseError(text=var_10, code=var_14, position=var_13)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_15) == 1
    var_16 = var_1.items()
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_16) == 1
    with pytest.raises(AssertionError):
        module_0.Message(text=var_13, index=var_11, position=var_13, end_position=var_4)

@pytest.mark.xfail(strict=True)
def test_case_40():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationError(text=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_0, index=var_3, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'HnePPWyN_ZR'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__str__()
    assert var_5 == "{'HnePPWyN_ZR': 'HnePPWyN_ZR'}"
    var_6 = 740
    var_7 = False
    var_8 = -28
    var_9 = module_0.Position(var_6, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 740
    assert var_9.column_no is False
    assert var_9.char_index == -28
    var_10 = var_9.__eq__(var_3)
    assert var_10 is False
    var_11 = None
    var_12 = var_4.__eq__(var_11)
    assert var_12 is False
    var_13 = module_0.ValidationResult()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert var_13.error is None
    var_14 = var_4.__repr__()
    assert var_14 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_15 = var_9.__repr__()
    assert var_15 == 'Position(line_no=740, column_no=False, char_index=-28)'
    module_0.ParseError()

@pytest.mark.xfail(strict=True)
def test_case_41():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = 740
    var_5 = module_0.Position(var_4, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 740
    assert var_5.column_no == 740
    assert var_5.char_index == 740
    var_6 = None
    var_7 = module_0.Position(var_6, var_0, var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no == 'HnePPWyN_ZR'
    assert var_7.char_index == 'HnePPWyN_ZR'
    var_8 = var_7.__eq__(var_5)
    assert var_8 is False
    var_9 = None
    var_10 = var_2.__eq__(var_6)
    assert var_10 is False
    var_11 = var_2.__hash__()
    assert var_11 == -6602527800710040010
    var_12 = module_0.ValidationResult()
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert var_12.error is None
    var_13 = var_2.__repr__()
    assert var_13 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_14 = None
    var_15 = var_7.__repr__()
    assert var_15 == 'Position(line_no=None, column_no=HnePPWyN_ZR, char_index=HnePPWyN_ZR)'
    module_0.ParseError(text=var_9, key=var_14)

@pytest.mark.xfail(strict=True)
def test_case_42():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.Message(text=var_0, key=var_1, end_position=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.Message'
    assert var_2.text == 'HnePPWyN_ZR'
    assert var_2.code == 'custom'
    assert f'{type(var_2.index).__module__}.{type(var_2.index).__qualname__}' == 'builtins.list'
    assert len(var_2.index) == 1
    assert var_2.start_position is None
    assert f'{type(var_2.end_position).__module__}.{type(var_2.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.end_position) == 1
    var_3 = var_1.__str__()
    assert var_3 == 'HnePPWyN_ZR'
    var_4 = 740
    var_5 = module_0.Position(var_4, var_4, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.Position'
    assert var_5.line_no == 740
    assert var_5.column_no == 740
    assert var_5.char_index == 740
    var_6 = var_5.__eq__(var_5)
    assert var_6 is True
    var_7 = var_2.__eq__(var_2)
    assert var_7 is True
    var_8 = module_0.ValidationResult()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value is None
    assert var_8.error is None
    var_9 = var_2.__repr__()
    assert var_9 == "Message(text='HnePPWyN_ZR', code='custom', index=[ValidationError(text='HnePPWyN_ZR', code='custom')])"
    var_10 = var_5.__repr__()
    assert var_10 == 'Position(line_no=740, column_no=740, char_index=740)'
    var_11 = var_10.__iter__()
    var_12 = module_0.ParseError(text=var_9, code=var_9, position=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ParseError'
    assert len(var_12) == 1
    var_13 = var_1.items()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'collections.abc.ItemsView'
    assert len(var_13) == 1
    var_14 = var_13.__repr__()
    assert var_14 == "ItemsView(ValidationError(text='HnePPWyN_ZR', code='custom'))"
    var_15 = var_1.__iter__()
    var_16 = var_12.__eq__(var_1)
    assert var_16 is False
    var_17 = var_1.__eq__(var_15)
    assert var_17 is False
    var_18 = var_1.__hash__()
    assert var_18 == 4817588873692346812
    var_19 = var_11.__iter__()
    var_13.__contains__(var_16)

@pytest.mark.xfail(strict=True)
def test_case_43():
    var_0 = 'HnePPWyN_ZR'
    var_1 = None
    var_2 = module_0.ValidationError(text=var_0, code=var_0, key=var_1, messages=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = module_0.Message(text=var_1, index=var_1, position=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.Message'
    assert var_3.text is None
    assert var_3.code == 'custom'
    assert var_3.index == []
    assert f'{type(var_3.start_position).__module__}.{type(var_3.start_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.start_position) == 1
    assert f'{type(var_3.end_position).__module__}.{type(var_3.end_position).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.end_position) == 1
    var_4 = var_2.__str__()
    assert var_4 == 'HnePPWyN_ZR'
    var_5 = 740
    var_6 = True
    var_7 = module_0.Position(var_1, var_6, var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.Position'
    assert var_7.line_no is None
    assert var_7.column_no is True
    assert var_7.char_index == 740
    var_8 = var_7.__eq__(var_0)
    assert var_8 is False
    var_9 = 844
    var_10 = module_0.Position(var_1, var_9, var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.Position'
    assert var_10.line_no is None
    assert var_10.column_no == 844
    assert var_10.char_index is True
    var_11 = var_7.__eq__(var_10)
    assert var_11 is False
    var_12 = None
    var_13 = var_3.__eq__(var_1)
    assert var_13 is False
    var_14 = module_0.ValidationResult()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_14.value is None
    assert var_14.error is None
    var_15 = var_3.__repr__()
    assert var_15 == "Message(text=None, code='custom', position=ValidationError(text='HnePPWyN_ZR', code='HnePPWyN_ZR'))"
    var_16 = module_0.ValidationResult(error=var_12)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_16.value is None
    assert var_16.error is None
    var_17 = var_7.__repr__()
    assert var_17 == 'Position(line_no=None, column_no=True, char_index=740)'
    module_0.ParseError(text=var_12)

@pytest.mark.xfail(strict=True)
def test_case_44():
    var_0 = 'HnePPWyN_ZR'
    var_1 = module_0.ValidationError(text=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_1) == 1
    var_2 = module_0.ValidationError(text=var_0, key=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2) == 1
    var_3 = None
    var_4 = module_0.Message(text=var_0, index=var_3, position=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.Message'
    assert var_4.text == 'HnePPWyN_ZR'
    assert var_4.code == 'custom'
    assert var_4.index == []
    assert var_4.start_position is None
    assert var_4.end_position is None
    var_5 = var_2.__str__()
    assert var_5 == "{'HnePPWyN_ZR': 'HnePPWyN_ZR'}"
    var_6 = 740
    var_7 = False
    var_8 = -28
    var_9 = module_0.Position(var_6, var_7, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.Position'
    assert var_9.line_no == 740
    assert var_9.column_no is False
    assert var_9.char_index == -28
    var_10 = var_9.__eq__(var_3)
    assert var_10 is False
    var_11 = None
    var_12 = var_1.__str__()
    assert var_12 == 'HnePPWyN_ZR'
    var_13 = var_4.__eq__(var_11)
    assert var_13 is False
    var_14 = var_1.__eq__(var_2)
    assert var_14 is False
    var_15 = module_0.ValidationResult()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert var_15.error is None
    var_16 = var_4.__repr__()
    assert var_16 == "Message(text='HnePPWyN_ZR', code='custom')"
    var_17 = var_9.__repr__()
    assert var_17 == 'Position(line_no=740, column_no=False, char_index=-28)'
    module_0.ParseError()