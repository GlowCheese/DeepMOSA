# Check out: https://github.com/GlowCheese/deepmosa
import pytest
import typesystem.fields as module_0
import typesystem.schemas as module_1
import typesystem.base as module_2

def test_case_0():
    var_0 = module_0.String()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.fields.String'
    assert var_0.title == ''
    assert var_0.description == ''
    assert var_0.allow_null is False
    assert var_0.read_only is False
    assert var_0.allow_blank is False
    assert var_0.trim_whitespace is True
    assert var_0.max_length is None
    assert var_0.min_length is None
    assert var_0.format is None
    assert var_0.coerce_types is True
    assert var_0.pattern is None
    assert var_0.pattern_regex is None
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.String.errors == {'type': 'Must be a string.', 'null': 'May not be null.', 'blank': 'Must not be blank.', 'max_length': 'Must have no more than {max_length} characters.', 'min_length': 'Must have at least {min_length} characters.', 'pattern': 'Must match the pattern /{pattern}/.', 'format': 'Must be a valid {format}.'}
    var_1 = {var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0, var_0: var_0}
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
    with pytest.raises(module_2.ValidationError):
        var_2.validate(var_1)

def test_case_1():
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

def test_case_2():
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
    var_2 = None
    with pytest.raises(module_2.ValidationError):
        var_1.validate(var_2)

def test_case_3():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.__setitem__(var_1, var_1)
    assert len(var_0) == 1
    var_3 = var_0.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_4():
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
def test_case_5():
    var_0 = 'MOK+p.(n^'
    var_1 = None
    var_2 = module_1.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == 'MOK+p.(n^'
    assert var_2.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = var_2.validate_or_error(var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_3.value is None
    assert f'{type(var_3.error).__module__}.{type(var_3.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_3.error) == 1
    var_4 = var_2.serialize(var_1)
    var_4.__iter__()

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = '|Yc^\x0bh0)kQ'
    var_1 = None
    var_2 = module_1.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == '|Yc^\x0bh0)kQ'
    assert var_2.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_2.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = [var_0, var_0]
    module_1.Definitions(*var_1)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = {}
    module_1.Schema(var_0, **var_1)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_1.Definitions()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_2) == 0
    var_3 = 'dbMcJI:x5~'
    var_4 = None
    var_5 = var_2.setdefault(var_4)
    assert len(var_2) == 1
    var_6 = {var_3: var_5}
    module_1.Schema(var_6)

def test_case_10():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = module_1.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(var_1.fields).__module__}.{type(var_1.fields).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1.fields) == 0
    assert var_1.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_2 = var_1.validate(var_0)

def test_case_11():
    var_0 = {}
    var_1 = module_1.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = None
    var_3 = var_1.__setitem__(var_2, var_2)
    assert len(var_1) == 1
    var_4 = module_1.Schema(var_0)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.fields == {}
    assert var_4.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = var_1.__len__()
    assert var_5 == 1
    var_6 = var_4.has_default()
    assert var_6 is False
    with pytest.raises(module_2.ValidationError):
        var_4.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_12():
    var_0 = 'e0OZ'
    var_1 = None
    var_2 = {}
    var_3 = module_1.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_1.Schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.fields == {}
    assert var_4.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = None
    var_6 = var_4.validate_or_error(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = 'M(OJ.!jScKtFhz/b7.'
    var_8 = {var_0: var_4, var_0: var_4, var_7: var_4}
    var_9 = module_1.Schema(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_9.title == ''
    assert var_9.description == ''
    assert var_9.allow_null is False
    assert var_9.read_only is False
    assert f'{type(var_9.fields).__module__}.{type(var_9.fields).__qualname__}' == 'builtins.dict'
    assert len(var_9.fields) == 2
    assert var_9.required == ['e0OZ', 'M(OJ.!jScKtFhz/b7.']
    var_10 = var_4.serialize(var_1)
    var_11 = var_4.serialize(var_2)
    var_12 = var_4.validate(var_3)
    var_3.__delitem__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = 'e0OZ'
    var_1 = module_1.Reference(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert var_1.to == 'e0OZ'
    assert var_1.definitions == 'e0OZ'
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_2 = {}
    var_3 = module_1.Definitions()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_3) == 0
    var_4 = module_1.Schema(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert var_4.fields == {}
    assert var_4.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = var_4.__or__(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.fields.Union'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.any_of).__module__}.{type(var_5.any_of).__qualname__}' == 'builtins.list'
    assert len(var_5.any_of) == 2
    assert module_0.Union.errors == {'null': 'May not be null.', 'union': 'Did not match any valid type.'}
    var_6 = var_4.validate(var_2)
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    var_7 = var_4.serialize(var_6)
    var_1.validate(var_7)

def test_case_14():
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
    var_2 = var_1.serialize(var_0)

@pytest.mark.xfail(strict=True)
def test_case_15():
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
    var_1.validate(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default == 'default_value'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_2 = {var_0: var_1}
    var_3 = module_1.Schema(var_2)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_3.title == ''
    assert var_3.description == ''
    assert var_3.allow_null is False
    assert var_3.read_only is False
    assert f'{type(var_3.fields).__module__}.{type(var_3.fields).__qualname__}' == 'builtins.dict'
    assert len(var_3.fields) == 1
    assert var_3.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3.validate(var_2)

@pytest.mark.xfail(strict=True)
def test_case_17():
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
    var_2 = None
    var_3 = var_1.serialize(var_2)
    var_4 = var_1.validate_or_error(var_2)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_4.value is None
    assert f'{type(var_4.error).__module__}.{type(var_4.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_4.error) == 1
    var_5 = var_1.serialize(var_0)
    var_5.get_default_value()

def test_case_18():
    var_0 = module_1.Definitions()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_0) == 0
    var_1 = None
    var_2 = var_0.setdefault(var_1)
    assert len(var_0) == 1
    var_3 = None
    with pytest.raises(AssertionError):
        var_0.__setitem__(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = '0)Z'
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
    var_3 = ')c9\n7\x0c\tNF5"w,Lo'
    var_4 = 'M(OJ.!jScKtFhz/b7.'
    var_5 = {var_3: var_2, var_3: var_2, var_4: var_2}
    var_6 = module_1.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == [')c9\n7\x0c\tNF5"w,Lo', 'M(OJ.!jScKtFhz/b7.']
    var_7 = var_6.serialize(var_0)
    var_8 = var_2.get_default_value()
    var_8.__contains__(var_3)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = {}
    var_1 = module_1.Definitions()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_1) == 0
    var_2 = module_1.Schema(var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.fields == {}
    assert var_2.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_3 = ')c9\n7\x0c\tNF5"w,Lo'
    var_4 = 'M(OJ.!jScKtFhz/b7.'
    var_5 = {var_3: var_2, var_3: var_2, var_4: var_2}
    var_6 = module_1.Schema(var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_6.title == ''
    assert var_6.description == ''
    assert var_6.allow_null is False
    assert var_6.read_only is False
    assert f'{type(var_6.fields).__module__}.{type(var_6.fields).__qualname__}' == 'builtins.dict'
    assert len(var_6.fields) == 2
    assert var_6.required == [')c9\n7\x0c\tNF5"w,Lo', 'M(OJ.!jScKtFhz/b7.']
    var_7 = var_2.serialize(var_0)
    var_8 = var_6.serialize(var_7)
    var_9 = var_2.validate(var_1)
    var_7.get_default_value()

def test_case_21():
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
    var_2 = ')c9\n7\x0c\tNF5"w,Lo'
    var_3 = 'M(OJ.!jScKtFhz/b7.'
    var_4 = {var_2: var_1, var_2: var_1, var_3: var_1}
    var_5 = module_1.Schema(var_4)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_5.title == ''
    assert var_5.description == ''
    assert var_5.allow_null is False
    assert var_5.read_only is False
    assert f'{type(var_5.fields).__module__}.{type(var_5.fields).__qualname__}' == 'builtins.dict'
    assert len(var_5.fields) == 2
    assert var_5.required == [')c9\n7\x0c\tNF5"w,Lo', 'M(OJ.!jScKtFhz/b7.']
    var_6 = var_1.get_default_value()
    with pytest.raises(module_2.ValidationError):
        var_5.validate(var_0)

def test_case_22():
    var_0 = 'default_value'
    var_1 = module_0.Field(default=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.default == 'default_value'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is False
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_2 = 'field_with_default'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = {}
    var_6 = var_4.validate(var_5)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = '0)@Z'
    var_1 = None
    var_2 = module_1.Reference(var_0, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'typesystem.schemas.Reference'
    assert var_2.title == ''
    assert var_2.description == ''
    assert var_2.allow_null is False
    assert var_2.read_only is False
    assert var_2.to == '0)@Z'
    assert var_2.definitions is None
    assert module_1.Reference.errors == {'null': 'May not be null.'}
    assert f'{type(module_1.Reference.target).__module__}.{type(module_1.Reference.target).__qualname__}' == 'builtins.property'
    var_3 = var_2.serialize(var_1)
    var_4 = {var_0: var_2, var_3: var_2}
    var_5 = module_1.Definitions()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_5) == 0
    var_6 = var_2.validate_or_error(var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_6.value is None
    assert f'{type(var_6.error).__module__}.{type(var_6.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_6.error) == 1
    var_7 = module_1.Schema(var_4)
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_7.title == ''
    assert var_7.description == ''
    assert var_7.allow_null is False
    assert var_7.read_only is False
    assert f'{type(var_7.fields).__module__}.{type(var_7.fields).__qualname__}' == 'builtins.dict'
    assert len(var_7.fields) == 2
    assert var_7.required == ['0)@Z', None]
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_8 = None
    var_9 = var_7.validate_or_error(var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'typesystem.base.ValidationResult'
    assert var_9.value is None
    assert f'{type(var_9.error).__module__}.{type(var_9.error).__qualname__}' == 'typesystem.base.ValidationError'
    assert len(var_9.error) == 1
    var_10 = var_5.__iter__()
    var_11 = {}
    var_12 = module_1.Definitions(**var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'typesystem.schemas.Definitions'
    assert len(var_12) == 0
    var_13 = var_5.__setitem__(var_8, var_1)
    assert len(var_5) == 1
    var_7.serialize(var_9)

def test_case_24():
    var_0 = True
    var_1 = module_0.Field(read_only=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'typesystem.fields.Field'
    assert var_1.title == ''
    assert var_1.description == ''
    assert var_1.allow_null is False
    assert var_1.read_only is True
    assert f'{type(module_0.NO_DEFAULT).__module__}.{type(module_0.NO_DEFAULT).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.FORMATS).__module__}.{type(module_0.FORMATS).__qualname__}' == 'builtins.dict'
    assert len(module_0.FORMATS) == 7
    assert module_0.Field.errors == {}
    var_2 = 'read_only_field'
    var_3 = {var_2: var_1}
    var_4 = module_1.Schema(var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'typesystem.schemas.Schema'
    assert var_4.title == ''
    assert var_4.description == ''
    assert var_4.allow_null is False
    assert var_4.read_only is False
    assert f'{type(var_4.fields).__module__}.{type(var_4.fields).__qualname__}' == 'builtins.dict'
    assert len(var_4.fields) == 1
    assert var_4.required == []
    assert module_1.Schema.errors == {'type': 'Must be an object.', 'null': 'May not be null.', 'invalid_key': 'All object keys must be strings.', 'required': 'This field is required.'}
    var_5 = 'other_field'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = var_4.validate(var_7)