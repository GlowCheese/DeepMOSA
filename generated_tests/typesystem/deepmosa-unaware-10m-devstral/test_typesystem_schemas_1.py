# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.schemas as module_0
import typesystem.base as module_1
import typesystem.fields as module_2

def test_case_0():
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
    var_2 = None
    with pytest.raises(module_1.ValidationError):
        var_1.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = var_1.__setitem__(var_0, var_0)
    assert len(var_1) == 1
    var_3 = var_1.__len__()
    assert var_3 == 1
    var_2.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = None
    var_1 = module_0.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.to is None
    assert var_1.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    var_1 = '\tu+`H#WN'
    var_2 = None
    var_3 = module_0.Reference(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == '\tu+`H#WN'
    assert var_3.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = var_3.serialize(var_0)
    var_4.keys()

def test_case_5():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'I`;6k.Lw9Wh"mz/'
    var_1 = None
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_0.Definitions(**var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 1
    var_4 = module_0.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = var_4.__len__()
    assert var_5 == 0
    var_6 = module_0.Definitions()
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_6) == 0
    var_3.__getitem__(var_1)

def test_case_7():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = var_0.__len__()
    assert var_2 == 0
    var_3 = module_0.Schema(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3.fields) == 0
    assert var_3.required == []

@pytest.mark.xfail(strict=True)
def test_case_8():
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
    var_2 = None
    var_3 = var_1.get_default_value()
    var_4 = module_0.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = var_1.validate_or_error(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == {}
    assert var_5.error is None
    var_6 = var_4.__setitem__(var_3, var_3)
    assert len(var_4) == 1
    var_7 = var_1.validate_or_error(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_8 = var_4.__delitem__(var_3)
    assert len(var_4) == 0
    var_9 = var_1.validate_or_error(var_4)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value == {}
    assert var_9.error is None
    var_10 = var_1.serialize(var_3)
    var_11 = var_1.serialize(var_0)
    var_12 = module_0.Reference(var_3, var_10)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert var_12.to is None
    assert var_12.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_13 = module_0.Reference(var_6, var_6)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.to is None
    assert var_13.definitions is None
    var_14 = var_13.serialize(var_2)
    var_13.validate(var_14)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = ')\t`%+*t w\x0b'
    var_1 = 'agA_'
    var_2 = module_0.Reference(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == ')\t`%+*t w\x0b'
    assert var_2.definitions == ')\t`%+*t w\x0b'
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 2
    assert var_4.required == [')\t`%+*t w\x0b', 'agA_']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {var_1: var_2}
    var_6 = None
    var_7 = var_4.validate_or_error(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_7.value is None
    assert f'{type(var_7.error).__module__}.{type(var_7.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_7.error) == 1
    var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
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
    var_3 = var_2.validate(var_1)
    var_3.__getitem__(var_0)

def test_case_11():
    var_0 = 'k:N,K+gc6[\x0c'
    var_1 = True
    var_2 = None
    var_3 = module_2.Field(title=var_0, allow_null=var_2, read_only=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == 'k:N,K+gc6[\x0c'
    assert var_3.description == ''
    assert var_3.allow_null is None
    assert var_3.read_only is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_4 = {var_3: var_3, var_0: var_3}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 2
    assert var_5.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_1.ValidationError):
        var_5.validate(var_0)

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
    var_2 = None
    var_3 = var_1.serialize(var_0)
    var_4 = var_1.serialize(var_2)

def test_case_13():
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
    var_2 = var_1.serialize(var_0)

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
    var_2 = None
    var_3 = None
    var_4 = module_0.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = var_1.validate_or_error(var_0)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value == {}
    assert var_5.error is None
    var_6 = var_4.__setitem__(var_3, var_3)
    assert len(var_4) == 1
    var_4.__setitem__(var_2, var_6)

def test_case_15():
    var_0 = '|I"avDX0o o'
    var_1 = module_2.Field(allow_null=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null == '|I"avDX0o o'
    assert var_1.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_0.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

def test_case_16():
    var_0 = 'aHKUVO%W e\\F]fSUO'
    var_1 = False
    var_2 = module_2.Field(read_only=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_3 = {var_0: var_2, var_0: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['aHKUVO%W e\\F]fSUO']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = ')\t`%+*t w\x0b'
    var_1 = 'agA_'
    var_2 = module_0.Reference(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == ')\t`%+*t w\x0b'
    assert var_2.definitions == ')\t`%+*t w\x0b'
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_1: var_2, var_1: var_2, var_0: var_2, var_1: var_2}
    var_4 = var_2.serialize(var_3)
    var_5 = None
    var_6 = var_2.serialize(var_5)
    var_7 = module_0.Schema(var_3)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 2
    assert var_7.required == ['agA_', ')\t`%+*t w\x0b']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = {var_1: var_2, var_1: var_6, var_0: var_2}
    var_9 = None
    var_10 = var_7.validate_or_error(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_10.value is None
    assert f'{type(var_10.error).__module__}.{type(var_10.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_10.error) == 1
    var_7.validate(var_8)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '|I"avDX0o o'
    var_1 = module_2.Field(allow_null=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null == '|I"avDX0o o'
    assert var_1.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_2 = {var_0: var_1, var_0: var_1}
    var_3 = module_0.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = True
    var_1 = module_2.Field(allow_null=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_2 = {var_1: var_1, var_1: var_1}
    var_3 = module_0.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = ')\t`%+t*t w\x0b'
    var_1 = 'agA_'
    var_2 = True
    var_3 = module_2.Field(allow_null=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.default is None
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is True
    assert var_3.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_4 = {var_0: var_3, var_1: var_3}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 2
    assert var_5.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = var_5.serialize(var_5)
    var_7 = {var_1: var_3}
    var_8 = None
    var_9 = var_5.validate_or_error(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert f'{type(var_9.error).__module__}.{type(var_9.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.error) == 1
    var_5.validate(var_7)

def test_case_21():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_2.Field()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_3 = module_2.Field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 2
    assert var_5.required == ['name', 'age']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = 'John'
    var_7 = 30
    var_8 = {var_0: var_6, var_1: var_7}
    var_9 = var_5.serialize(var_8)
    var_10 = var_5.serialize(var_9)
    var_11 = var_5.serialize(var_9)
    var_12 = 'address'
    var_13 = module_2.Field()
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Field'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    var_14 = 'city'
    var_15 = module_2.Field()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Field'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    var_16 = {var_14: var_15}
    var_17 = module_0.Schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.fields).__module__}.{type(var_17.fields).__qualname__}' == 'builtins.dict'
    assert len(var_17.fields) == 1
    assert var_17.required == ['city']
    var_18 = {var_0: var_13, var_12: var_17}
    var_19 = module_0.Schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.fields).__module__}.{type(var_19.fields).__qualname__}' == 'builtins.dict'
    assert len(var_19.fields) == 2
    assert var_19.required == ['name', 'address']
    var_20 = var_19.serialize(var_11)
    var_21 = var_5.serialize(var_8)

def test_case_22():
    var_0 = 'agA_'
    var_1 = module_0.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.to == 'agA_'
    assert var_1.definitions == 'agA_'
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_2 = {var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1, var_0: var_1}
    var_3 = None
    var_4 = module_0.Schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['agA_']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {var_0: var_1, var_0: var_3, var_3: var_1, var_3: var_3}
    with pytest.raises(module_1.ValidationError):
        var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_2 = None
    var_3 = var_1.validate_or_error(var_0)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value == {}
    assert var_3.error is None
    var_4 = var_0.__setitem__(var_2, var_2)
    assert var_1.fields == {None: None}
    var_5 = var_1.validate_or_error(var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1
    var_6 = var_1.validate_or_error(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_1.serialize(var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = '|I"avDX0o o'
    var_1 = False
    var_2 = module_2.Field(allow_null=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_3 = {var_0: var_2, var_0: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['|I"avDX0o o']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4.validate(var_3)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = '3|e'
    var_1 = 'S"S'
    var_2 = module_2.Field(allow_null=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null == '3|e'
    assert var_2.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 2
    assert var_4.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {var_1: var_2}
    var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = ''
    var_1 = 'flND=J'
    var_2 = module_2.Field(allow_null=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null == ''
    assert var_2.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_3 = {var_0: var_2, var_1: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 2
    assert var_4.required == ['', 'flND=J']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {var_1: var_2}
    var_4.validate(var_5)

def test_case_27():
    var_0 = ')\t`%+*t w\x0b'
    var_1 = 'a}A|_'
    var_2 = True
    var_3 = None
    var_4 = module_2.Field(allow_null=var_3, read_only=var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is None
    assert var_4.read_only is True
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_5 = {var_0: var_4, var_1: var_4}
    var_6 = module_0.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = {var_1: var_4}
    var_8 = None
    var_9 = var_6.validate_or_error(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert f'{type(var_9.error).__module__}.{type(var_9.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.error) == 1
    var_10 = var_6.validate(var_7)
    var_11 = var_6.validate_or_error(var_0)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_11.value is None
    assert f'{type(var_11.error).__module__}.{type(var_11.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_11.error) == 1