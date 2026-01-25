# Check out: https://github.com/GlowCheese/deepmosa
import enum as module_2

import pyrsistent._checked_types as module_1
import pyrsistent._field_common as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.serialize(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = None
    module_0.is_type_cls(var_0, var_0)

def test_case_2():
    var_0 = None
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

def test_case_3():
    var_0 = module_0.field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_4():
    var_0 = None
    module_0.field(initial=var_0, factory=var_0, serializer=var_0)

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = None
    module_0.field(invariant=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.pvector_field(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.pmap_field(var_0, var_0)

def test_case_8():
    pass

def test_case_9():
    var_0 = None
    var_1 = module_0.PTypeError(var_0, var_0, var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common.PTypeError'
    assert var_1.source_class is None
    assert var_1.field is None
    assert var_1.expected_types is None
    assert var_1.actual_type is None
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.pset_field(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.field(initial=var_0, serializer=var_0)

def test_case_12():
    var_0 = module_0.field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = '__reducWR_'
    module_0.field(var_0, mandatory=var_0)

def test_case_14():
    var_0 = module_0.field()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_1 = module_0.check_type(var_0, var_0, var_0, var_0)

def test_case_15():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = False
    var_1 = 'failure_code'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = [var_3]
    module_0.check_global_invariants(var_3, var_4)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = 'TestClass'
    module_0.set_fields(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = '__reducWR_'
    module_0.pmap_field(var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = '__reduIWR_'
    var_1 = None
    module_0.is_type_cls(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    module_0.set_fields(var_1, var_1, var_1)

def test_case_21():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

def test_case_22():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.pvector_field(var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'

def test_case_23():
    var_0 = lambda self: True
    var_1 = lambda : None
    var_2 = lambda x: x
    var_3 = module_0.field(initial=var_1, mandatory=var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.pmap_field(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    module_0.pvector_field(var_1, invariant=var_0, item_invariant=var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    module_0.pmap_field(var_0, var_0, var_0, var_0)

def test_case_26():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.field(var_0, mandatory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    module_0.field(var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    module_0.pvector_field(var_0, var_0, invariant=var_0)

@pytest.mark.xfail(strict=True)
def test_case_29():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.is_type_cls(var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_2 = module_0.field(var_0, mandatory=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_3 = var_2.__ge__(var_2)
    var_4 = module_0.is_field_ignore_extra_complaint(var_2, var_2, var_2)
    assert var_4 is False
    module_0.field(invariant=var_3, factory=var_2)

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = None
    var_2 = module_0.pvector_field(var_0, var_0, var_1, item_invariant=var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_3 = None
    module_0.field(invariant=var_0, factory=var_3, serializer=var_0)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.field(var_0, mandatory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    module_0.check_type(var_1, var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_32():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = var_0.__eq__(var_0)
    assert var_1 is True
    var_2 = module_0.field(var_0, mandatory=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_1.wrap_invariant(var_0)
    var_4 = module_0.field(invariant=var_3, serializer=var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._field_common._PField'
    module_0.field(var_1, var_3, mandatory=var_3)

@pytest.mark.xfail(strict=True)
def test_case_33():
    var_0 = module_1.optional()
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    var_1 = module_0.field(var_0, mandatory=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_2 = None
    var_3 = module_0.is_field_ignore_extra_complaint(var_1, var_1, var_2)
    assert var_3 is False
    var_4 = module_0.check_type(var_0, var_1, var_0, var_2)
    module_0.serialize(var_2, var_0, var_2)

def test_case_34():
    var_0 = {}
    var_1 = ()
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_4 = bool('fields' in var_0)
    assert var_4 is True
    var_5 = var_0['fields']

@pytest.mark.xfail(strict=True)
def test_case_35():
    var_0 = []
    var_1 = None
    var_2 = module_0.is_field_ignore_extra_complaint(var_1, var_0, var_1)
    assert var_2 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    module_0._restore_seq_field_pickle(var_1, var_0, var_2)