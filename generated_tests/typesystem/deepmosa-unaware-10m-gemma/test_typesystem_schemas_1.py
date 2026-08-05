# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.schemas as module_0
import typesystem.base as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = "V' 9;Z%!>."
    var_1 = module_0.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.to == "V' 9;Z%!>."
    assert var_1.definitions == "V' 9;Z%!>."
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_2 = {var_0: var_1}
    var_3 = module_0.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ["V' 9;Z%!>."]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

def test_case_1():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = var_2.serialize(var_0)
    var_4 = None
    module_0.Schema(var_4)

def test_case_3():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = "V' 9;Z%!>."
    var_3 = module_0.Reference(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == "V' 9;Z%!>."
    assert var_3.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = {var_2: var_3}
    var_5 = module_0.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = module_0.Schema(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ["V' 9;Z%!>."]
    var_7 = var_3.serialize(var_4)
    var_8 = var_1.validate_or_error(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_8.value == {}
    assert var_8.error is None
    var_9 = var_1.serialize(var_7)
    var_10 = var_6.validate_or_error(var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert f'{type(var_10.error).__module__}.{type(var_10.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10.error) == 1
    var_11 = var_6.serialize(var_9)
    var_12 = var_5.__setitem__(var_3, var_5)
    assert len(var_5) == 1

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = ''
    var_2 = module_0.Reference(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == ''
    assert var_2.definitions == ''
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_2.validate(var_0)

def test_case_5():
    var_0 = None
    var_1 = "V' 9;Z%!>."
    var_2 = module_0.Reference(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == "V' 9;Z%!>."
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ["V' 9;Z%!>."]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = var_2.serialize(var_0)

def test_case_6():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = var_0.__len__()
    assert var_1 == 0
    var_2 = var_0.__len__()
    assert var_2 == 0

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = 'i'
    var_1 = None
    var_2 = 'M/oq[ '
    var_3 = {var_0: var_1, var_0: var_1, var_2: var_1}
    var_4 = module_0.Definitions(**var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 2
    var_4.__getitem__(var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = {}
    var_1 = "V' 9;Z%!>."
    var_2 = module_0.Reference(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == "V' 9;Z%!>."
    assert var_2.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_1: var_2}
    var_4 = module_0.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = module_0.Schema(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ["V' 9;Z%!>."]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = var_5.validate_or_error(var_0)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_4.popitem()

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = None
    var_0.__delitem__(var_2)

def test_case_10():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_1.ValidationError):
        var_2.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = "V' 9;Z%!>."
    var_3 = module_0.Reference(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == "V' 9;Z%!>."
    assert var_3.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = {var_2: var_3}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ["V' 9;Z%!>."]
    var_6 = var_3.serialize(var_4)
    var_5.validate(var_6)

def test_case_12():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = var_1.validate(var_0)

def test_case_13():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = {}
    var_2 = module_0.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = var_2.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_14():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = 'V" ;;%!.'
    var_3 = module_0.Reference(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == 'V" ;;%!.'
    assert var_3.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = {var_2: var_3, var_2: var_3, var_2: var_1}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['V" ;;%!.']
    var_6 = var_3.serialize(var_4)
    var_7 = var_5.validate_or_error(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = None
    var_6.__setitem__(var_6, var_8)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = {}
    var_3 = var_0.setdefault(var_1)
    assert len(var_0) == 1
    var_4 = module_0.Schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.fields == {}
    assert var_4.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = var_4.serialize(var_4)
    var_0.__setitem__(var_1, var_3)

def test_case_16():
    var_0 = {}
    var_1 = "V' 9;Z%!>."
    var_2 = module_0.Reference(var_1, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == "V' 9;Z%!>."
    assert var_2.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_1: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ["V' 9;Z%!>."]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = var_4.validate_or_error(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_17():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = "V' 9;Z%!>."
    var_3 = module_0.Reference(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == "V' 9;Z%!>."
    assert var_3.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = {var_2: var_3, var_2: var_1}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ["V' 9;Z%!>."]
    var_6 = var_3.serialize(var_4)
    var_7 = None
    var_8 = var_5.serialize(var_7)
    var_9 = var_6.__setitem__(var_7, var_6)
    var_10 = var_3.validate_or_error(var_7)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert f'{type(var_10.error).__module__}.{type(var_10.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10.error) == 1
    var_11 = var_5.validate_or_error(var_7)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert f'{type(var_11.error).__module__}.{type(var_11.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11.error) == 1
    with pytest.raises(module_1.ValidationError):
        var_5.validate(var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = "V' 9;Z%!>."
    var_3 = module_0.Reference(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == "V' 9;Z%!>."
    assert var_3.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = {var_2: var_3}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ["V' 9;Z%!>."]
    var_6 = var_3.serialize(var_4)
    var_7 = var_1.validate_or_error(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value == {}
    assert var_7.error is None
    var_8 = var_3.has_default()
    assert var_8 is False
    var_9 = var_5.serialize(var_8)
    var_9.validate_or_error(var_6)

def test_case_19():
    var_0 = "m'+;"
    var_1 = ''
    var_2 = None
    var_3 = [var_1]
    var_4 = module_0.Definitions(*var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = module_0.Reference(var_2, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to is None
    assert f'{type(var_5.definitions).__module__}.{type(var_5.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5.definitions) == 0
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = None
    var_7 = module_2.Field(description=var_1, default=var_6, allow_null=var_6, read_only=var_1)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Field'
    assert var_7.default is None
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is None
    assert var_7.read_only == ''
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_8 = {var_0: var_7, var_0: var_7}
    var_9 = module_0.Schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 1
    assert var_9.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = {}
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.Schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = module_0.Schema(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    var_4 = var_1.__len__()
    assert var_4 == 0
    var_5 = module_0.Reference(var_4, var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == 0
    assert var_5.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = {var_4: var_5}
    var_7 = module_0.Definitions()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_7) == 0
    var_8 = module_0.Schema(var_6)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 1
    assert var_8.required == [0]
    var_9 = var_5.serialize(var_6)
    var_10 = var_2.validate_or_error(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert f'{type(var_10.error).__module__}.{type(var_10.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10.error) == 1
    var_11 = None
    var_12 = var_9.__setitem__(var_3, var_11)
    var_13 = var_8.validate_or_error(var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_13.value is None
    assert f'{type(var_13.error).__module__}.{type(var_13.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_13.error) == 1
    var_8.serialize(var_10)

def test_case_21():
    var_0 = {}
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = "V' 9;Z%!>."
    var_3 = module_0.Reference(var_2, var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == "V' 9;Z%!>."
    assert var_3.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = {var_2: var_3, var_2: var_1}
    var_5 = module_0.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = module_0.Schema(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ["V' 9;Z%!>."]
    var_7 = var_3.serialize(var_4)
    var_8 = var_5.keys()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'collections.abc.KeysView'
    assert len(var_8) == 0
    var_9 = var_1.validate_or_error(var_7)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == {}
    assert var_9.error is None
    var_10 = var_6.serialize(var_7)
    var_11 = None
    var_12 = var_6.validate_or_error(var_0)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_12.value is None
    assert f'{type(var_12.error).__module__}.{type(var_12.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_12.error) == 1
    var_13 = var_6.serialize(var_0)
    var_14 = var_5.__setitem__(var_11, var_7)
    assert len(var_5) == 1
    assert len(var_8) == 1
    var_15 = var_3.validate_or_error(var_11)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_15.value is None
    assert f'{type(var_15.error).__module__}.{type(var_15.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_15.error) == 1
    var_16 = module_0.Reference(var_14, var_14)
    assert f'{type(var_16).__module__}.{type(var_16).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_16.title == ''
    assert var_16.description == ''
    assert var_16.allow_null is False
    assert var_16.read_only is False
    assert var_16.to is None
    assert var_16.definitions is None
    var_17 = var_6.validate_or_error(var_11)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_17.value is None
    assert f'{type(var_17.error).__module__}.{type(var_17.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_17.error) == 1
    var_18 = var_6.validate(var_10)