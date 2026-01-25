# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import mimesis.exceptions as module_0
import mimesis.enums as module_1

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.AliasesTypeError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    var_1 = module_0.SchemaError()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.SchemaError'
    var_2 = var_1.__str__()
    assert var_2 == 'The schema must be a callable object that returns a dict.See https://mimesis.name/en/master/schema.html for more details.'
    var_3 = var_1.__str__()
    assert var_3 == 'The schema must be a callable object that returns a dict.See https://mimesis.name/en/master/schema.html for more details.'
    var_4 = var_1.__str__()
    assert var_4 == 'The schema must be a callable object that returns a dict.See https://mimesis.name/en/master/schema.html for more details.'
    var_5 = module_1.Locale.DA
    var_6 = module_0.LocaleError(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_6.locale == module_1.Locale.DA
    var_7 = var_0.__str__()
    assert var_7 == "The 'aliases' attribute needs to be a non-nested dictionary where keys are the aliases and values are the corresponding field names."
    var_8 = var_6.__str__()
    assert var_8 == 'Invalid locale «Locale.DA»'
    var_9 = module_0.AliasesTypeError()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    module_0.NonEnumerableError(var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    var_1 = module_0.NonEnumerableError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.NonEnumerableError'
    assert var_1.items == ''
    assert module_0.NonEnumerableError.message == 'You should use one item of: «{}» of the object mimesis.enums.{}'
    var_1.__str__()

def test_case_2():
    var_0 = None
    var_1 = module_0.LocaleError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_1.locale is None
    var_2 = var_1.__str__()
    assert var_2 == 'Invalid locale «None»'
    var_3 = None
    var_4 = module_0.FieldError()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_4.name is None
    assert var_4.message == 'Field «{}» is not supported.'
    assert var_4.message_none == 'The field cannot have the value None.'
    var_5 = module_0.FieldError(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_5.name is None
    assert var_5.message == 'Field «{}» is not supported.'
    assert var_5.message_none == 'The field cannot have the value None.'
    var_6 = var_5.__str__()
    assert var_6 == 'The field cannot have the value None.'
    var_7 = var_1.__str__()
    assert var_7 == 'Invalid locale «None»'

def test_case_3():
    var_0 = module_0.FieldsetError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    var_1 = 'K'
    var_2 = var_0.__str__()
    assert var_2 == 'The «iterations» parameter should be greater than 1.'
    var_3 = module_0.FieldError(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.FieldError'
    assert var_3.name == 'K'
    assert var_3.message == 'Field «{}» is not supported.'
    assert var_3.message_none == 'The field cannot have the value None.'
    var_4 = var_3.__str__()
    assert var_4 == 'Field «K» is not supported.'
    var_5 = None
    var_6 = module_0.FieldNameError()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_6.name is None
    var_7 = var_6.__str__()
    assert var_7 == 'The field name «None» is not a valid Python identifier.'
    var_8 = module_0.FieldNameError(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_8.name is None
    var_9 = module_0.LocaleError(var_5)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_9.locale is None
    var_10 = var_9.__str__()
    assert var_10 == 'Invalid locale «None»'
    var_11 = var_9.__str__()
    assert var_11 == 'Invalid locale «None»'
    var_12 = var_3.__str__()
    assert var_12 == 'Field «K» is not supported.'
    var_13 = var_3.__str__()
    assert var_13 == 'Field «K» is not supported.'
    var_14 = module_0.AliasesTypeError()
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    var_15 = module_0.FieldNameError(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_15.name == 'Invalid locale «None»'
    var_16 = var_3.__str__()
    assert var_16 == 'Field «K» is not supported.'
    var_17 = var_8.__str__()
    assert var_17 == 'The field name «None» is not a valid Python identifier.'
    var_18 = module_0.AliasesTypeError()
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    var_19 = module_0.FieldArityError()
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'mimesis.exceptions.FieldArityError'
    var_20 = var_14.__str__()
    assert var_20 == "The 'aliases' attribute needs to be a non-nested dictionary where keys are the aliases and values are the corresponding field names."
    var_21 = var_19.__str__()
    assert var_21 == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"
    var_22 = module_0.SchemaError()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'mimesis.exceptions.SchemaError'
    var_23 = var_9.__str__()
    assert var_23 == 'Invalid locale «None»'
    var_24 = var_18.__str__()
    assert var_24 == "The 'aliases' attribute needs to be a non-nested dictionary where keys are the aliases and values are the corresponding field names."
    var_25 = var_9.__str__()
    assert var_25 == 'Invalid locale «None»'
    var_26 = module_0.AliasesTypeError()
    assert f'{type(var_26).__module__}.{type(var_26).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    var_27 = module_0.FieldNameError()
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_27.name is None
    var_28 = var_27.__str__()
    assert var_28 == 'The field name «None» is not a valid Python identifier.'

def test_case_4():
    var_0 = module_0.SchemaError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.SchemaError'
    var_1 = var_0.__str__()
    assert var_1 == 'The schema must be a callable object that returns a dict.See https://mimesis.name/en/master/schema.html for more details.'

def test_case_5():
    var_0 = module_0.FieldArityError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldArityError'
    var_1 = var_0.__str__()
    assert var_1 == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"
    var_2 = var_0.__str__()
    assert var_2 == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"
    var_3 = module_0.LocaleError(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.LocaleError'
    assert var_3.locale == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"
    var_4 = module_0.FieldArityError()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'mimesis.exceptions.FieldArityError'
    var_5 = module_0.SchemaError()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.exceptions.SchemaError'
    var_6 = var_4.__str__()
    assert var_6 == "The custom handler must accept at least two arguments: 'random' and '**kwargs'"

def test_case_6():
    var_0 = module_0.FieldNameError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_0.name is None
    var_1 = module_0.FieldsetError()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    var_2 = var_1.__str__()
    assert var_2 == 'The «iterations» parameter should be greater than 1.'
    var_3 = module_0.FieldsetError()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    var_4 = var_3.__str__()
    assert var_4 == 'The «iterations» parameter should be greater than 1.'
    var_5 = module_0.FieldNameError()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_5.name is None
    var_6 = var_3.__str__()
    assert var_6 == 'The «iterations» parameter should be greater than 1.'
    var_7 = module_0.FieldsetError()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'mimesis.exceptions.FieldsetError'
    var_8 = var_7.__str__()
    assert var_8 == 'The «iterations» parameter should be greater than 1.'

def test_case_7():
    var_0 = None
    var_1 = module_0.FieldNameError(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_1.name is None
    var_2 = module_0.FieldNameError()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_2.name is None

def test_case_8():
    var_0 = module_0.FieldNameError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.FieldNameError'
    assert var_0.name is None
    var_1 = var_0.__str__()
    assert var_1 == 'The field name «None» is not a valid Python identifier.'
    var_2 = var_0.__str__()
    assert var_2 == 'The field name «None» is not a valid Python identifier.'
    var_3 = module_0.SchemaError()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.exceptions.SchemaError'
    var_4 = var_3.__str__()
    assert var_4 == 'The schema must be a callable object that returns a dict.See https://mimesis.name/en/master/schema.html for more details.'

def test_case_9():
    var_0 = module_0.AliasesTypeError()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.exceptions.AliasesTypeError'
    var_1 = var_0.__str__()
    assert var_1 == "The 'aliases' attribute needs to be a non-nested dictionary where keys are the aliases and values are the corresponding field names."