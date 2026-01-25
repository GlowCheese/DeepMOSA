# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = {var_0: var_0, var_0: var_0}
    var_2 = module_1.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(var_2.fields).__module__}.{type(var_2.fields).__qualname__}' == 'builtins.dict'
    assert len(var_2.fields) == 1
    assert f'{type(var_2.required).__module__}.{type(var_2.required).__qualname__}' == 'builtins.list'
    assert len(var_2.required) == 1
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_1.setdefault(var_2)
    assert len(var_1) == 1
    module_1.Schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    var_1 = module_1.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.to is None
    assert var_1.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_1.validate(var_0)

@pytest.mark.xfail(strict=True)
def test_case_3():
    var_0 = {}
    var_1 = None
    var_2 = module_1.Reference(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to is None
    assert var_2.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = var_2.serialize(var_1)
    var_4 = module_1.Definitions(**var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_3.get_default_value()

def test_case_4():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0

def test_case_5():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_0.popitem()

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'dFbMcJI:x5~'
    var_2 = var_0.__len__()
    assert var_2 == 0
    var_3 = {var_1: var_2, var_2: var_2}
    module_1.Schema(var_3)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_1.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = None
    var_1.__delitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    var_1 = False
    var_2 = module_1.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to is None
    assert var_2.definitions is False
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_2.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'G2#r\x0c\x0cp,;Vs-#qO'
    var_1 = None
    var_2 = None
    var_3 = module_1.Reference(var_0, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == 'G2#r\x0c\x0cp,;Vs-#qO'
    assert var_3.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = var_3.validate_or_error(var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_3.validate(var_4)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = var_0.__iter__()
    var_2 = None
    var_3 = var_0.setdefault(var_2)
    assert len(var_0) == 1
    var_0.__setitem__(var_2, var_3)

def test_case_12():
    var_0 = None
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_2.ValidationError):
        var_2.validate(var_0)

def test_case_13():
    var_0 = {}
    var_1 = module_1.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = var_1.validate(var_0)

def test_case_14():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = var_0.__len__()
    assert var_1 == 0
    var_2 = module_1.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = '71vBb0\rL**m&V'
    var_4 = {}
    var_5 = module_1.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.fields == {}
    assert var_5.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_2.ValidationError):
        var_5.validate(var_3)

def test_case_15():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = {}
    var_2 = module_1.Schema(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = None
    var_4 = module_1.Reference(var_2, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.to).__module__}.{type(var_4.to).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_5 = var_2.validate(var_1)
    var_6 = var_2.serialize(var_3)

def test_case_16():
    var_0 = {}
    var_1 = module_1.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.fields == {}
    assert var_1.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = var_1.validate(var_0)
    var_3 = var_1.serialize(var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = module_0.Field()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.fields.Field'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_3 = module_0.Field()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 2
    assert var_5.required == ['name', 'age']
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = True
    var_7 = module_0.Field(read_only=var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Field'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is True
    var_8 = 'id'
    var_9 = module_0.Field()
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.fields.Field'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    var_10 = {var_8: var_7, var_0: var_9}
    var_11 = module_1.Schema(var_10)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 2
    assert var_11.required == ['name']
    var_12 = 'default_value'
    var_13 = module_0.Field(default=var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.fields.Field'
    assert var_13.default == 'default_value'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    var_14 = 'status'
    var_15 = module_0.Field()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Field'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    var_16 = {var_14: var_13, var_0: var_15}
    var_17 = module_1.Schema(var_16)
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    assert f'{type(var_17.fields).__module__}.{type(var_17.fields).__qualname__}' == 'builtins.dict'
    assert len(var_17.fields) == 2
    assert var_17.required == ['name']
    var_18 = 'email'
    var_19 = module_0.Field(read_only=var_6)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.fields.Field'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is True
    var_20 = 'active'
    var_21 = module_0.Field(default=var_20)
    assert f'{type(var_21).__module__}.{type(var_21).__qualname__}' == 'typesystem.fields.Field'
    assert var_21.default == 'active'
    assert var_21.title == ''
    assert var_21.description == ''
    assert var_21.allow_null is False
    assert var_21.read_only is False
    var_22 = module_0.Field()
    assert f'{type(var_22).__module__}.{type(var_22).__qualname__}' == 'typesystem.fields.Field'
    assert var_22.title == ''
    assert var_22.description == ''
    assert var_22.allow_null is False
    assert var_22.read_only is False
    var_23 = module_0.Field()
    assert f'{type(var_23).__module__}.{type(var_23).__qualname__}' == 'typesystem.fields.Field'
    assert var_23.title == ''
    assert var_23.description == ''
    assert var_23.allow_null is False
    assert var_23.read_only is False
    var_24 = {var_8: var_19, var_14: var_21, var_0: var_22, var_18: var_23}
    var_25 = module_1.Schema(var_24)
    assert f'{type(var_25).__module__}.{type(var_25).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_25.title == ''
    assert var_25.description == ''
    assert var_25.allow_null is False
    assert var_25.read_only is False
    assert f'{type(var_25.fields).__module__}.{type(var_25.fields).__qualname__}' == 'builtins.dict'
    assert len(var_25.fields) == 4
    assert var_25.required == ['name', 'email']
    var_26 = {}
    var_27 = module_1.Schema(var_26)
    assert f'{type(var_27).__module__}.{type(var_27).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_27.title == ''
    assert var_27.description == ''
    assert var_27.allow_null is False
    assert var_27.read_only is False
    assert var_27.fields == {}
    assert var_27.required == []
    var_17.validate(var_10)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = 'ke~?B'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_1.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['ke~?B']
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = var_3.serialize(var_2)
    var_3.validate(var_4)

def test_case_19():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = 'Uymt*/D'
    var_2 = {var_1: var_0}
    var_3 = module_1.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['Uymt*/D']
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = {var_1: var_0, var_1: var_3}
    var_5 = module_1.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['Uymt*/D']
    var_6 = var_5.serialize(var_2)
    with pytest.raises(module_2.ValidationError):
        var_5.validate(var_6)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = 'ke~?B'
    var_2 = {var_1: var_0, var_1: var_0}
    var_3 = module_1.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == ['ke~?B']
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = module_0.Field()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    var_2 = module_1.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'key2'
    var_4 = {var_3: var_1, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == ['key2']
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_6: var_6, var_3: var_7}
    var_9 = var_5.serialize(var_8)
    var_10 = module_0.Field()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.fields.Field'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    var_11 = module_0.Field()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Field'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    var_12 = {var_10: var_10, var_3: var_11}
    var_13 = module_1.Schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.fields).__module__}.{type(var_13.fields).__qualname__}' == 'builtins.dict'
    assert len(var_13.fields) == 2
    assert f'{type(var_13.required).__module__}.{type(var_13.required).__qualname__}' == 'builtins.list'
    assert len(var_13.required) == 2
    var_14 = var_13.serialize(var_8)
    var_15 = module_0.Field()
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.fields.Field'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    var_13.serialize(var_10)

def test_case_22():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.Field'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_1 = 'key1'
    var_2 = {}
    var_3 = module_1.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.fields == {}
    assert var_3.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4 = module_0.Field()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.fields.Field'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    var_5 = 'key2'
    var_6 = {var_1: var_4, var_5: var_4}
    var_7 = module_1.Schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 2
    assert var_7.required == ['key1', 'key2']
    var_8 = 'vjlue2'
    var_9 = {var_1: var_5, var_5: var_8}
    var_10 = var_7.serialize(var_9)
    var_11 = {var_1: var_0, var_5: var_7}
    var_12 = module_1.Schema(var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.fields).__module__}.{type(var_12.fields).__qualname__}' == 'builtins.dict'
    assert len(var_12.fields) == 2
    assert var_12.required == ['key1', 'key2']
    var_13 = {var_1: var_3, var_5: var_3}
    var_14 = module_1.Schema(var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_14.title == ''
    assert var_14.description == ''
    assert var_14.allow_null is False
    assert var_14.read_only is False
    assert f'{type(var_14.fields).__module__}.{type(var_14.fields).__qualname__}' == 'builtins.dict'
    assert len(var_14.fields) == 2
    assert var_14.required == ['key1', 'key2']
    var_15 = var_12.serialize(var_10)
    var_16 = var_14.serialize(var_15)
    var_17 = module_0.Field()
    assert f'{type(var_17).__module__}.{type(var_17).__qualname__}' == 'typesystem.fields.Field'
    assert var_17.title == ''
    assert var_17.description == ''
    assert var_17.allow_null is False
    assert var_17.read_only is False
    var_18 = {var_1: var_17}
    var_19 = module_1.Schema(var_18)
    assert f'{type(var_19).__module__}.{type(var_19).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_19.title == ''
    assert var_19.description == ''
    assert var_19.allow_null is False
    assert var_19.read_only is False
    assert f'{type(var_19.fields).__module__}.{type(var_19.fields).__qualname__}' == 'builtins.dict'
    assert len(var_19.fields) == 1
    assert var_19.required == ['key1']
    var_20 = var_14.validate(var_16)