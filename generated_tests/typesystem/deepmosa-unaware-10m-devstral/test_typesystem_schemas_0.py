# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.schemas as module_0
import typesystem.base as module_1
import typesystem.fields as module_2

def test_case_0():
    var_0 = 'BVhhQ:`\t$C'
    var_1 = None
    var_2 = module_0.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == 'BVhhQ:`\t$C'
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_0: var_2, var_0: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['BVhhQ:`\t$C']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}

def test_case_1():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.__setitem__(var_1, var_1)
    assert len(var_0) == 1

@pytest.mark.xfail(strict=True)
def test_case_2():
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
    var_2 = var_1.serialize(var_0)
    var_1.validate(var_0)

def test_case_4():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0

def test_case_5():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_0.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.to).__module__}.{type(var_1.to).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1.to) == 0
    assert f'{type(var_1.definitions).__module__}.{type(var_1.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1.definitions) == 0
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_2 = {var_1: var_1, var_1: var_1, var_1: var_1}
    var_3 = module_0.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert f'{type(var_3.required).__module__}.{type(var_3.required).__qualname__}' == 'builtins.list'
    assert len(var_3.required) == 1
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_1.ValidationError):
        var_3.validate(var_0)

def test_case_6():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'BVhhQ:`\t$C'
    var_2 = var_0.__len__()
    assert var_2 == 0
    var_3 = None
    var_4 = module_0.Reference(var_1, var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.to == 'BVhhQ:`\t$C'
    assert var_4.definitions == 0
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_5 = var_4.validate_or_error(var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_5.value is None
    assert f'{type(var_5.error).__module__}.{type(var_5.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_5.error) == 1

def test_case_7():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = '#YazdynT}Bh.s3M\\w~M'
    var_2 = None
    var_3 = var_0.__setitem__(var_2, var_2)
    assert len(var_0) == 1
    var_4 = var_0.clear()
    assert len(var_0) == 0
    var_5 = module_0.Reference(var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == '#YazdynT}Bh.s3M\\w~M'
    assert var_5.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = var_5.validate_or_error(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = 'lD}yR_nj4cOH'
    var_8 = {var_7: var_5, var_4: var_5, var_7: var_5}
    var_9 = module_0.Schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 2
    assert var_9.required == ['lD}yR_nj4cOH', None]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_1.ValidationError):
        var_9.validate(var_0)

def test_case_8():
    var_0 = '%'
    var_1 = None
    var_2 = module_0.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == '%'
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.__setitem__(var_1, var_1)
    assert len(var_0) == 1
    var_0.__setitem__(var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = 'BVhhQ:`\t$C'
    var_1 = None
    var_2 = module_0.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == 'BVhhQ:`\t$C'
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_0: var_2, var_0: var_2, var_0: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['BVhhQ:`\t$C']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_4.validate_or_error(var_3)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    var_1 = None
    var_2 = module_0.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to is None
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_2.serialize(var_2)

def test_case_12():
    var_0 = 'BVhhQ:`\t$C'
    var_1 = None
    var_2 = module_0.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == 'BVhhQ:`\t$C'
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = {var_0: var_2, var_0: var_2}
    var_4 = module_0.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == ['BVhhQ:`\t$C']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_1.ValidationError):
        var_4.validate(var_1)

def test_case_13():
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
    var_3 = var_1.__setitem__(var_0, var_0)
    assert len(var_1) == 1
    var_4 = {var_3: var_2, var_3: var_2}
    var_5 = module_0.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 1
    assert var_5.required == [None]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    with pytest.raises(module_1.ValidationError):
        var_5.validate(var_1)

def test_case_14():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'BVhhQ:`\t$C'
    var_2 = None
    var_3 = module_0.Reference(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == 'BVhhQ:`\t$C'
    assert var_3.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = 'lD}yR_nj4cOH'
    var_6 = ']J4%jA'
    var_7 = {var_5: var_3, var_6: var_3}
    var_8 = module_0.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 2
    assert var_8.required == ['lD}yR_nj4cOH', ']J4%jA']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = var_0.__iter__()
    with pytest.raises(module_1.ValidationError):
        var_8.validate(var_9)

def test_case_15():
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
    var_2 = var_1.validate_or_error(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_2.value is None
    assert f'{type(var_2.error).__module__}.{type(var_2.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_2.error) == 1
    var_3 = 'lD}yR_nj4cOH'
    var_4 = ']J4%jA'
    var_5 = {var_3: var_1, var_4: var_1}
    var_6 = module_0.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == ['lD}yR_nj4cOH', ']J4%jA']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = var_6.serialize(var_0)
    var_8 = module_0.Schema(var_5)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 2
    assert var_8.required == ['lD}yR_nj4cOH', ']J4%jA']
    with pytest.raises(module_1.ValidationError):
        var_8.validate(var_7)

def test_case_16():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'BVhhQ:`\t$C'
    var_2 = None
    var_3 = module_0.Reference(var_1, var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert var_3.to == 'BVhhQ:`\t$C'
    assert var_3.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_4 = var_3.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = {}
    var_6 = module_0.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert var_6.fields == {}
    assert var_6.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = var_3.serialize(var_2)
    var_8 = var_7.__hash__()
    assert var_8 == 8444165580952
    var_9 = var_6.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = None
    var_1 = '#YazdynT}Bh.s3M\\w~M'
    var_2 = None
    var_3 = None
    var_4 = module_0.Definitions()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_4) == 0
    var_5 = module_0.Reference(var_1, var_2)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to == '#YazdynT}Bh.s3M\\w~M'
    assert var_5.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = var_5.validate_or_error(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = var_4.__len__()
    assert var_7 == 0
    var_8 = var_4.__setitem__(var_3, var_3)
    assert len(var_4) == 1
    var_9 = {var_0: var_5, var_0: var_5, var_0: var_5}
    var_10 = module_0.Definitions()
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_10) == 0
    var_11 = module_0.Schema(var_9)
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    assert f'{type(var_11.fields).__module__}.{type(var_11.fields).__qualname__}' == 'builtins.dict'
    assert len(var_11.fields) == 1
    assert var_11.required == [None]
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_11.serialize(var_10)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = module_0.Reference(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to is None
    assert var_2.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = 'lD}yR_nj4cOH'
    var_5 = ']J4%jA'
    var_6 = {var_4: var_2, var_5: var_2}
    var_7 = module_0.Schema(var_6)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 2
    assert var_7.required == ['lD}yR_nj4cOH', ']J4%jA']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = var_7.serialize(var_2)
    var_9 = module_0.Schema(var_6)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 2
    assert var_9.required == ['lD}yR_nj4cOH', ']J4%jA']
    var_10 = var_0.__contains__(var_1)
    assert var_10 is False
    var_10.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = module_0.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = 'BVhhQ:`\t$C'
    var_2 = None
    var_3 = None
    var_4 = None
    var_5 = module_0.Reference(var_3, var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert var_5.to is None
    assert var_5.definitions is None
    assert module_0.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_0.Reference.target).__module__}.{type(module_0.Reference.target).__qualname__}' == 'builtins.property'
    var_6 = var_5.validate_or_error(var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = 'lD}yR_nj4cOH'
    var_8 = ']J4%jA'
    var_9 = {var_7: var_5, var_8: var_5}
    var_10 = module_0.Schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.fields).__module__}.{type(var_10.fields).__qualname__}' == 'builtins.dict'
    assert len(var_10.fields) == 2
    assert var_10.required == ['lD}yR_nj4cOH', ']J4%jA']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_11 = var_10.serialize(var_4)
    var_12 = module_0.Schema(var_9)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_12.title == ''
    assert var_12.description == ''
    assert var_12.allow_null is False
    assert var_12.read_only is False
    assert f'{type(var_12.fields).__module__}.{type(var_12.fields).__qualname__}' == 'builtins.dict'
    assert len(var_12.fields) == 2
    assert var_12.required == ['lD}yR_nj4cOH', ']J4%jA']
    var_13 = module_0.Reference(var_1, var_0)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert var_13.to == 'BVhhQ:`\t$C'
    assert f'{type(var_13.definitions).__module__}.{type(var_13.definitions).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_13.definitions) == 0
    var_14 = var_0.__setitem__(var_4, var_11)
    assert len(var_0) == 1
    assert len(var_13.definitions) == 1
    var_12.serialize(var_9)

def test_case_20():
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
    var_5 = True
    var_6 = module_0.Schema(var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == ['name', 'age']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_7 = module_2.Field(read_only=var_5)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Field'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is True
    var_8 = module_2.Field()
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Field'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_0.Schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.fields).__module__}.{type(var_10.fields).__qualname__}' == 'builtins.dict'
    assert len(var_10.fields) == 2
    assert var_10.required == ['age']
    var_11 = module_2.Field()
    assert f'{type(var_11).__module__}.{type(var_11).__qualname__}' == 'typesystem.fields.Field'
    assert var_11.title == ''
    assert var_11.description == ''
    assert var_11.allow_null is False
    assert var_11.read_only is False
    var_12 = {var_0: var_3, var_1: var_11}
    var_13 = module_0.Schema(var_12)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.fields).__module__}.{type(var_13.fields).__qualname__}' == 'builtins.dict'
    assert len(var_13.fields) == 2
    assert var_13.required == ['name', 'age']
    var_14 = {}
    var_15 = module_0.Schema(var_14)
    assert f'{type(var_15).__module__}.{type(var_15).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_15.title == ''
    assert var_15.description == ''
    assert var_15.allow_null is False
    assert var_15.read_only is False
    assert var_15.fields == {}
    assert var_15.required == []

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_2.Field(allow_null=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_4 = True
    var_5 = module_0.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = module_2.Field(allow_null=var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.fields.Field'
    assert var_6.default is None
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is True
    assert var_6.read_only is False
    var_7 = {var_0: var_3, var_1: var_6}
    var_8 = module_0.Schema(var_7)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is False
    assert var_8.read_only is False
    assert f'{type(var_8.fields).__module__}.{type(var_8.fields).__qualname__}' == 'builtins.dict'
    assert len(var_8.fields) == 2
    assert var_8.required == ['name']
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_9 = 30
    var_10 = {var_0: var_1, var_1: var_9}
    var_8.validate(var_10)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = False
    var_3 = module_2.Field(allow_null=var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.fields.Field'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(module_2.NO_DEFAULT).__module__}.{type(module_2.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_2.FORMATS).__module__}.{type(module_2.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_2.FORMATS) == 7
    assert module_2.Field.errors == {}
    var_4 = True
    var_5 = module_2.Field(allow_null=var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Field'
    assert var_5.default is None
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is True
    assert var_5.read_only is False
    var_6 = 'Unknown'
    var_7 = module_2.Field(default=var_6, allow_null=var_2)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.fields.Field'
    assert var_7.default == 'Unknown'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    var_8 = module_2.Field(default=var_2, allow_null=var_4)
    assert f'{type(var_8).__module__}.{type(var_8).__qualname__}' == 'typesystem.fields.Field'
    assert var_8.default is False
    assert var_8.title == ''
    assert var_8.description == ''
    assert var_8.allow_null is True
    assert var_8.read_only is False
    var_9 = {var_0: var_7, var_1: var_8}
    var_10 = module_0.Schema(var_9)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_10.title == ''
    assert var_10.description == ''
    assert var_10.allow_null is False
    assert var_10.read_only is False
    assert f'{type(var_10.fields).__module__}.{type(var_10.fields).__qualname__}' == 'builtins.dict'
    assert len(var_10.fields) == 2
    assert var_10.required == []
    assert module_0.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_11 = {}
    var_12 = var_10.validate(var_11)
    var_13 = module_0.Schema(var_9)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_13.title == ''
    assert var_13.description == ''
    assert var_13.allow_null is False
    assert var_13.read_only is False
    assert f'{type(var_13.fields).__module__}.{type(var_13.fields).__qualname__}' == 'builtins.dict'
    assert len(var_13.fields) == 2
    assert var_13.required == []
    var_13.validate(var_12)
    assert var_14 is None