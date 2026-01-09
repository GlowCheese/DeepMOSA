# Check out: https://github.com/GlowCheese/deepmosa
import tokenize as module_2

import mimesis.exceptions as module_1
import mimesis.schema as module_0
import pytest


def test_case_0():
    var_0 = 693
    var_1 = module_0.SchemaContext(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaContext'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.SchemaContext.custom).__module__}.{type(module_0.SchemaContext.custom).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.index).__module__}.{type(module_0.SchemaContext.index).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.iteration).__module__}.{type(module_0.SchemaContext.iteration).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.schema_builder).__module__}.{type(module_0.SchemaContext.schema_builder).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.seed).__module__}.{type(module_0.SchemaContext.seed).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.timestamp).__module__}.{type(module_0.SchemaContext.timestamp).__qualname__}' == 'builtins.member_descriptor'

def test_case_1():
    var_0 = 691
    var_1 = module_0.SchemaContext(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaContext'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.SchemaContext.custom).__module__}.{type(module_0.SchemaContext.custom).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.index).__module__}.{type(module_0.SchemaContext.index).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.iteration).__module__}.{type(module_0.SchemaContext.iteration).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.schema_builder).__module__}.{type(module_0.SchemaContext.schema_builder).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.seed).__module__}.{type(module_0.SchemaContext.seed).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.timestamp).__module__}.{type(module_0.SchemaContext.timestamp).__qualname__}' == 'builtins.member_descriptor'
    var_2 = '|)\\d\rFZ2Q0ox+")X'
    with pytest.raises(ValueError):
        var_1.pick_from(var_2, var_2)

def test_case_2():
    var_0 = module_0.SchemaBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

def test_case_3():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

def test_case_4():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'

def test_case_5():
    var_0 = None
    with pytest.raises(module_1.SchemaError):
        module_0.Schema(var_0)

def test_case_6():
    var_0 = module_0.SchemaBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.create()

def test_case_7():
    var_0 = None
    var_1 = module_0.SchemaBuilder(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = var_1.define(var_0, var_0)

def test_case_8():
    var_0 = False
    var_1 = module_0.SchemaBuilder(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'

def test_case_9():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.unregister_all_handlers()

def test_case_10():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.__str__()
    assert var_1 == 'BaseField <Locale.EN>'

def test_case_11():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    with pytest.raises(module_1.FieldError):
        var_0.perform()

def test_case_12():
    var_0 = module_0.SchemaBuilder()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = ';_\x0cK(`vyqc}iG+F$\x0b'
    var_2 = True
    var_3 = {var_1: var_2}
    with pytest.raises(ValueError):
        var_0.create(**var_3)

def test_case_13():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.get_random_instance()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.random.Random'
    assert var_1.gauss_next is None

def test_case_14():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = var_0.handle()

def test_case_15():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = ''
    var_2 = var_0.unregister_handler(var_1)

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_0.perform(var_0)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    module_2.detect_encoding(var_0)

def test_case_18():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = 'nB'
    with pytest.raises(module_1.FieldError):
        var_0.perform(var_1)

def test_case_19():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = var_0.__str__()
    assert var_1 == 'Fieldset <Locale.EN>'
    with pytest.raises(module_1.FieldError):
        var_0.perform(var_1)

def test_case_20():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.reseed()

def test_case_21():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = var_0.unregister_handlers()

def test_case_22():
    var_0 = module_0.BaseField()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = 'd>j?g@jM%z@\\P_C\nRAPF'
    with pytest.raises(module_1.FieldError):
        var_0.perform(var_1)

def test_case_23():
    var_0 = 693
    var_1 = module_0.SchemaContext(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaContext'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.SchemaContext.custom).__module__}.{type(module_0.SchemaContext.custom).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.index).__module__}.{type(module_0.SchemaContext.index).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.iteration).__module__}.{type(module_0.SchemaContext.iteration).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.schema_builder).__module__}.{type(module_0.SchemaContext.schema_builder).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.seed).__module__}.{type(module_0.SchemaContext.seed).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.timestamp).__module__}.{type(module_0.SchemaContext.timestamp).__qualname__}' == 'builtins.member_descriptor'
    with pytest.raises(ValueError):
        var_1.ref(var_0)

def test_case_24():
    var_0 = []
    var_1 = module_0.Fieldset(*var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_2 = ')4Hk'
    var_3 = var_1.unregister_handlers(var_2)

def test_case_25():
    var_0 = None
    var_1 = False
    with pytest.raises(ValueError):
        module_0.Schema(var_0, var_1)

def test_case_26():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    with pytest.raises(TypeError):
        var_0.register_handler(var_0, var_0)

def test_case_27():
    var_0 = 693
    var_1 = module_0.SchemaContext(var_0, custom=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaContext'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert f'{type(module_0.SchemaContext.custom).__module__}.{type(module_0.SchemaContext.custom).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.index).__module__}.{type(module_0.SchemaContext.index).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.iteration).__module__}.{type(module_0.SchemaContext.iteration).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.schema_builder).__module__}.{type(module_0.SchemaContext.schema_builder).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.seed).__module__}.{type(module_0.SchemaContext.seed).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0.SchemaContext.timestamp).__module__}.{type(module_0.SchemaContext.timestamp).__qualname__}' == 'builtins.member_descriptor'

def test_case_28():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = 'RUQ=p"nah\tk'
    with pytest.raises(module_1.FieldNameError):
        var_0.register_handler(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_0.Field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Field'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1 = 'd%'
    var_2 = None
    var_3 = {var_1: var_2}
    var_0.__call__(**var_3)

def test_case_30():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = 'MY'
    var_2 = var_0.register_handler(var_1, var_0)

def test_case_31():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = 'MY'
    with pytest.raises(TypeError):
        var_0.register_handler(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = 'nB'
    var_1 = module_0.BaseField()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.BaseField'
    assert f'{type(var_1.seed).__module__}.{type(var_1.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_1.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_1.register_handlers(var_0)

def test_case_33():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'

def test_case_34():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = None
    var_2 = var_0.handle(var_1)
    var_3 = 'MY'
    with pytest.raises(module_1.FieldArityError):
        var_0.register_handler(var_3, var_2)

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = 'MY'
    var_2 = var_0.register_handler(var_1, var_0)
    var_0.perform(var_1)

@pytest.mark.xfail(strict=True)
def test_case_36():
    var_0 = None
    var_1 = module_0.SchemaBuilder()
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    var_2 = '>j?g@jM%z@\\P_CRAPF'
    var_3 = var_1.define(var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'mimesis.schema.SchemaBuilder'
    var_4 = {var_2: var_0, var_2: var_0, var_2: var_0, var_2: var_0, var_2: var_0, var_2: var_0, var_2: var_0, var_2: var_0}
    var_1.create(**var_4)

@pytest.mark.xfail(strict=True)
def test_case_37():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'
    var_1.create()

@pytest.mark.xfail(strict=True)
def test_case_38():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'
    var_1.__next__()

def test_case_39():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'
    var_2 = var_1.map(var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.schema.Schema'

def test_case_40():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'
    var_2 = var_1.with_context()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.schema.Schema'

def test_case_41():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'
    var_2 = var_1.__iter__()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.schema.Schema'

def test_case_42():
    var_0 = module_0.Fieldset()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'mimesis.schema.Fieldset'
    assert f'{type(var_0.seed).__module__}.{type(var_0.seed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert var_0.aliases == {}
    assert f'{type(module_0.MissingSeed).__module__}.{type(module_0.MissingSeed).__qualname__}' == 'mimesis.types._MissingSeed'
    assert f'{type(module_0.Seed).__module__}.{type(module_0.Seed).__qualname__}' == 'types.UnionType'
    assert module_0.Fieldset.fieldset_default_iterations == 10
    assert module_0.Fieldset.fieldset_iterations_kwarg == 'i'
    var_1 = module_0.Schema(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'mimesis.schema.Schema'
    assert f'{type(module_0.Schema.iterations).__module__}.{type(module_0.Schema.iterations).__qualname__}' == 'builtins.member_descriptor'
    var_2 = var_1.iterator()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'mimesis.schema.Schema'