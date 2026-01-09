# Check out: https://github.com/GlowCheese/deepmosa
import mimesis.enums as module_1
import mimesis.exceptions as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    var_1 = module_0.LocaleError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_1.locale is None
    var_2 = var_1.__str__()
    assert var_2 == 'Invalid locale «None»'
    var_3 = module_0.FieldError()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_3.name is None
    assert var_3.message == 'Field «{}» is not supported.'
    assert var_3.message_none == 'The field cannot have the value None.'
    var_4 = var_3.__str__()
    assert var_4 == 'The field cannot have the value None.'
    module_0.NonEnumerableError(var_1)

def test_case_1():
    var_0 = None
    var_1 = module_0.NonEnumerableError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.NonEnumerableError'
    assert var_1.items == ''
    assert module_0.NonEnumerableError.message == 'You should use one item of: «{}» of the object mimesis.enums.{}'

def test_case_2():
    var_0 = module_0.FieldNameError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_0.name is None
    var_1 = var_0.__str__()
    assert var_1 == 'The field name «None» is not a valid Python identifier.'
    var_2 = module_0.FieldError()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_2.name is None
    assert var_2.message == 'Field «{}» is not supported.'
    assert var_2.message_none == 'The field cannot have the value None.'
    var_3 = var_2.__str__()
    assert var_3 == 'The field cannot have the value None.'
    var_4 = module_0.LocaleError(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_4.locale == 'The field name «None» is not a valid Python identifier.'
    var_5 = var_0.__str__()
    assert var_5 == 'The field name «None» is not a valid Python identifier.'
    var_6 = module_0.FieldNameError()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_6.name is None

def test_case_3():
    var_0 = module_0.SchemaError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.SchemaError'
    var_1 = var_0.__str__()
    assert var_1 == 'The schema must be a callable object that returns a dict.See https://mimesis.name/en/master/schema.html for more details.'

def test_case_4():
    var_0 = module_1.Locale.FA
    var_1 = var_0.__str__()
    assert var_1 == 'Locale.FA'
    var_2 = module_0.LocaleError(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_2.locale == module_1.Locale.FA
    var_3 = var_2.__str__()
    assert var_3 == 'Invalid locale «Locale.FA»'

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    var_1 = module_0.NonEnumerableError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.NonEnumerableError'
    assert var_1.items == ''
    assert module_0.NonEnumerableError.message == 'You should use one item of: «{}» of the object mimesis.enums.{}'
    var_1.__str__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '4dQ"D_B\rkq[$'
    var_1 = b''
    var_2 = module_0.FieldsetError()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    var_3 = var_2.__str__()
    assert var_3 == 'The «iterations» parameter should be greater than 1.'
    var_4 = {var_0: var_0, var_0: var_1}
    var_5 = module_0.FieldsetError()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    module_0.AliasesTypeError(**var_4)

def test_case_7():
    var_0 = None
    var_1 = module_0.FieldNameError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_1.name is None
    var_2 = var_1.__str__()
    assert var_2 == 'The field name «None» is not a valid Python identifier.'
    var_3 = module_0.NonEnumerableError(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.NonEnumerableError'
    assert var_3.items == ''
    assert module_0.NonEnumerableError.message == 'You should use one item of: «{}» of the object mimesis.enums.{}'

def test_case_8():
    var_0 = module_0.FieldArityError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldArityError'
    var_1 = var_0.__str__()
    assert var_1 == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"

def test_case_9():
    var_0 = module_0.AliasesTypeError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    var_1 = None
    var_2 = [var_1, var_1, var_1, var_1]
    var_3 = module_0.FieldArityError(*var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.FieldArityError'
    var_4 = var_3.__str__()
    assert var_4 == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"
    var_5 = var_0.__str__()
    assert var_5 == "The 'aliases' attribute needs to be a non-nested dictionary where keys are the aliases and values are the corresponding field names."

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = module_0.FieldsetError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    var_1 = var_0.__str__()
    assert var_1 == 'The «iterations» parameter should be greater than 1.'
    var_2 = None
    var_3 = module_0.NonEnumerableError(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.NonEnumerableError'
    assert var_3.items == ''
    assert module_0.NonEnumerableError.message == 'You should use one item of: «{}» of the object mimesis.enums.{}'
    var_4 = var_0.__str__()
    assert var_4 == 'The «iterations» parameter should be greater than 1.'
    var_5 = var_0.__str__()
    assert var_5 == 'The «iterations» parameter should be greater than 1.'
    var_6 = module_0.FieldError(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_6.name == 'The «iterations» parameter should be greater than 1.'
    assert var_6.message == 'Field «{}» is not supported.'
    assert var_6.message_none == 'The field cannot have the value None.'
    var_7 = var_0.__str__()
    assert var_7 == 'The «iterations» parameter should be greater than 1.'
    var_8 = var_6.__str__()
    assert var_8 == 'Field «The «iterations» parameter should be greater than 1.» is not supported.'
    var_9 = var_6.__str__()
    assert var_9 == 'Field «The «iterations» parameter should be greater than 1.» is not supported.'
    var_10 = None
    var_11 = [var_10, var_10, var_10]
    var_12 = module_0.FieldError(var_8)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_12.name == 'Field «The «iterations» parameter should be greater than 1.» is not supported.'
    assert var_12.message == 'Field «{}» is not supported.'
    assert var_12.message_none == 'The field cannot have the value None.'
    var_13 = module_0.FieldArityError(*var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'mimesis.exceptions.FieldArityError'
    var_14 = None
    var_15 = '.ie9Q9?<F*'
    var_16 = {var_15: var_14, var_1: var_14}
    module_0.FieldArityError(**var_16)