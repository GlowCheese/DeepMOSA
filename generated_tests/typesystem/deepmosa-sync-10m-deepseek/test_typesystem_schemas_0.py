# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.schemas as module_0
import typesystem.fields as module_1
import typesystem.base as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_1.setdefault(var_2)
    assert len(var_1) == 1
    var_4 = {var_3: var_3}
    module_0.Schema(var_4)

def test_case_1():
    var_0 = module_1.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
    var_2 = module_0.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'builtins.dict'
    assert len(var_2.fields) == 1
    assert f'{type(var_2.required).__module__}.{type(var_2.required).__qualname__}' == 'builtins.list'
    assert len(var_2.required) == 1
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = {var_0: var_0, var_0: var_2}
    var_4 = {}
    var_5 = module_0.Schema(var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert f'{type(var_5.required).__module__}.{type(var_5.required).__qualname__}' == 'builtins.list'
    assert len(var_5.required) == 1
    with pytest.raises(module_2.ValidationError):
        var_5.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.setdefault(var_1)
    assert len(var_0) == 1
    var_0.__contains__(var_0)

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
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.Reference(var_0, var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to is None
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = var_2.serialize(var_0)
    var_2.validate(var_0)

def test_case_5():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.Schema(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_0.popitem()

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_0.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'dbMcJI:x5~'
    var_4 = var_2.__len__()
    assert var_4 == 0
    var_5 = {var_3: var_4}
    module_0.Schema(var_5)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = module_0.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_1.__delitem__(var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    var_1.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = None
    var_2 = None
    var_3 = module_0.Reference(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to is None
    assert var_3.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = 'Xrq9R}#`4OPC~Z'
    var_5 = {var_4: var_3}
    var_6 = module_0.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 1
    assert var_6.required == ['Xrq9R}#`4OPC~Z']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = var_3.serialize(var_0)
    var_8 = var_3.serialize(var_5)
    var_9 = module_0.Reference(var_4, var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert var_9.to == 'Xrq9R}#`4OPC~Z'
    assert f'{type(var_9.definitions).__module__}.{type(var_9.definitions).__qualname__}' == 'builtins.dict'
    assert len(var_9.definitions) == 1
    var_3.validate(var_0)

def test_case_12():
    var_0 = True
    var_1 = module_1.Field(allow_null=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is True
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_2 = {}
    var_3 = 'allow_null'
    var_4 = {var_3: var_0}
    var_5 = module_0.Schema(var_2, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert var_5.fields == {}
    assert var_5.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None

def test_case_13():
    var_0 = 'default_value'
    var_1 = module_1.Field(default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default == 'default_value'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_2 = 'with_default'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Schema(var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = bool(var_7 == {'with_default': 'default_value'})
    assert var_8 is True

def test_case_14():
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
    var_2 = {}
    var_3 = module_0.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_2.ValidationError):
        var_3.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_1.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = var_0.__or__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.Schema(var_2, **var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['name']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5.validate(var_2)

def test_case_16():
    var_0 = 'default_value'
    var_1 = module_1.Field(default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default == 'default_value'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_2 = {}
    var_3 = module_0.Schema(var_2, **var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = {}
    var_5 = var_3.validate(var_4)
    var_6 = bool(var_5 == {'with_default': 'default_value'})

def test_case_17():
    var_0 = module_1.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_1 = 'name'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = var_0.__or__(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Union'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.any_of).__module__}.{type(var_4.any_of).__qualname__}' == 'builtins.list'
    assert len(var_4.any_of) == 2
    assert module_1.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_5 = module_0.Schema(var_2, **var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['name']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = var_5.serialize(var_3)
    with pytest.raises(module_2.ValidationError):
        var_5.validate(var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = None
    var_1 = 'Xrq9R}\x0b`4OPC~Z`G'
    var_2 = module_1.Field(default=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.default is None
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = module_0.Schema(var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = var_5.serialize(var_2)
    var_3.validate(var_3)

def test_case_19():
    var_0 = module_1.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_1 = 'required_field'
    var_2 = {var_1: var_0, var_1: var_0, var_1: var_0}
    var_3 = {}
    var_4 = module_0.Schema(var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['required_field']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {}
    with pytest.raises(module_2.ValidationError):
        var_4.validate(var_5)

def test_case_20():
    var_0 = module_1.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_1 = 'valid'
    var_2 = {var_1: var_0}
    var_3 = {}
    var_4 = module_0.Schema(var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['valid']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = None
    var_6 = var_4.serialize(var_5)
    var_7 = 1
    var_8 = 'value'
    var_9 = {var_7: var_8}
    with pytest.raises(module_2.ValidationError):
        var_4.validate(var_9)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = None
    var_1 = module_1.Field(default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default is None
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Schema(var_2, **var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = module_0.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_4.serialize(var_1)

def test_case_22():
    var_0 = True
    var_1 = module_1.Field(read_only=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is True
    assert f'{type(module_1.NO_DEFAULT).__module__}.{type(module_1.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_1.FORMATS).__module__}.{type(module_1.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_1.FORMATS) == 7
    assert module_1.Field.errors == {}
    var_2 = 'read_only'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Schema(var_3, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = var_5.validate(var_4)
    var_7 = bool(var_6 == {})
    assert var_7 is True

def test_case_23():
    var_0 = None
    var_1 = None
    var_2 = module_0.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = var_2.__setitem__(var_1, var_0)
    assert len(var_2) == 1
    with pytest.raises(AssertionError):
        var_2.__setitem__(var_0, var_1)

def test_case_24():
    var_0 = {}
    var_1 = 'some_ref'
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Reference(var_1, var_0, **var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    assert var_5.to == 'some_ref'
    assert var_5.definitions == {}
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = None
    var_7 = var_5.validate(var_6)
    assert var_7 is None