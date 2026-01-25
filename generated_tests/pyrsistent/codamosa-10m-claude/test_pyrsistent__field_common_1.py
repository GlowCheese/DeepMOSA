# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_3
import builtins as module_2
import enum as module_4
import tokenize as module_5

import pyrsistent._checked_types as module_1
import pyrsistent._field_common as module_0
import pytest


@pytest.mark.xfail(strict=True)
def test_case_0():
    var_0 = None
    module_0.serialize(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_1():
    var_0 = 'O->UD$Fn&I;Ps8 '
    var_1 = None
    module_0.is_type_cls(var_1, var_0)

@pytest.mark.xfail(strict=True)
def test_case_2():
    var_0 = None
    module_0.is_type_cls(var_0, var_0)

def test_case_3():
    var_0 = None
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

def test_case_4():
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
def test_case_5():
    var_0 = None
    module_0.field(invariant=var_0)

def test_case_6():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_1 = module_0.pmap_field(var_0, var_0, var_0)
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

def test_case_7():
    var_0 = None
    var_1 = module_0.field(initial=var_0)
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
def test_case_8():
    var_0 = None
    module_0.field(initial=var_0, mandatory=var_0, serializer=var_0)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.pset_field(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.pmap_field(var_0, var_0, invariant=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
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
    var_2 = module_0.field(initial=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_2.__missing__(var_2)

@pytest.mark.xfail(strict=True)
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
    module_0.pvector_field(var_0)

@pytest.mark.xfail(strict=True)
def test_case_13():
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
    module_0.field(initial=var_0, factory=var_0)

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
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False

def test_case_15():
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

@pytest.mark.xfail(strict=True)
def test_case_16():
    var_0 = {}
    var_1 = ()
    var_2 = '9cI'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_4 = set()
    module_0.check_global_invariants(var_3, var_3)

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = module_2.object()
    module_0.check_global_invariants(var_7, var_6)

@pytest.mark.xfail(strict=True)
def test_case_18():
    var_0 = module_3.Param
    var_1 = None
    module_0.pvector_field(var_0, initial=var_1)

def test_case_19():
    var_0 = []
    var_1 = module_0.check_global_invariants(var_0, var_0)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = module_1.CheckedPVector()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_0) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    module_0.serialize(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_21():
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
    var_1 = module_0.field(initial=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    var_2 = module_1.CheckedPVector()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_2) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_3 = module_0.pmap_field(var_2, var_2, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._field_common._PField'
    module_0.check_global_invariants(var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = {}
    var_1 = ()
    var_2 = '9cI'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_4 = '_pfields'
    var_5 = set()
    var_6 = '_pfields'
    module_0.set_fields(var_5, var_6, var_4)

@pytest.mark.xfail(strict=True)
def test_case_23():
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
    var_1 = var_0.__repr__()
    module_0.field(var_1)

def test_case_24():
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
    var_1 = set()
    var_2 = var_0.type
    var_3 = len(var_2)
    var_4 = var_0.type
    var_5 = len(var_4)
    var_6 = var_0.type
    var_7 = len(var_6)
    var_8 = True
    var_9 = module_0.field(mandatory=var_8)
    assert f'{type(var_9).__module__}.{type(var_9).__qualname__}' == 'pyrsistent._field_common._PField'
    var_10 = 2
    var_11 = lambda x: x * var_10
    var_12 = module_0.field(factory=var_11)
    assert f'{type(var_12).__module__}.{type(var_12).__qualname__}' == 'pyrsistent._field_common._PField'
    var_13 = lambda fmt, val: str(val)
    var_14 = module_0.field(serializer=var_13)
    assert f'{type(var_14).__module__}.{type(var_14).__qualname__}' == 'pyrsistent._field_common._PField'
    var_15 = 0
    var_16 = 'must be positive'
    var_17 = lambda x: (x > var_15, var_16)
    var_18 = module_0.field(invariant=var_17)
    assert f'{type(var_18).__module__}.{type(var_18).__qualname__}' == 'pyrsistent._field_common._PField'
    with pytest.raises(AttributeError):
        var_19 = var_11.invariant

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_4._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.is_type_cls(var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_2 = 'test_name'
    module_0.set_fields(var_0, var_1, var_2)

@pytest.mark.xfail(strict=True)
def test_case_26():
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
    var_1 = module_5.group()
    assert var_1 == '()'
    assert module_5.BOM_UTF8 == b'\xef\xbb\xbf'
    assert module_5.tok_name == {0: 'ENDMARKER', 1: 'NAME', 2: 'NUMBER', 3: 'STRING', 4: 'NEWLINE', 5: 'INDENT', 6: 'DEDENT', 7: 'LPAR', 8: 'RPAR', 9: 'LSQB', 10: 'RSQB', 11: 'COLON', 12: 'COMMA', 13: 'SEMI', 14: 'PLUS', 15: 'MINUS', 16: 'STAR', 17: 'SLASH', 18: 'VBAR', 19: 'AMPER', 20: 'LESS', 21: 'GREATER', 22: 'EQUAL', 23: 'DOT', 24: 'PERCENT', 25: 'LBRACE', 26: 'RBRACE', 27: 'EQEQUAL', 28: 'NOTEQUAL', 29: 'LESSEQUAL', 30: 'GREATEREQUAL', 31: 'TILDE', 32: 'CIRCUMFLEX', 33: 'LEFTSHIFT', 34: 'RIGHTSHIFT', 35: 'DOUBLESTAR', 36: 'PLUSEQUAL', 37: 'MINEQUAL', 38: 'STAREQUAL', 39: 'SLASHEQUAL', 40: 'PERCENTEQUAL', 41: 'AMPEREQUAL', 42: 'VBAREQUAL', 43: 'CIRCUMFLEXEQUAL', 44: 'LEFTSHIFTEQUAL', 45: 'RIGHTSHIFTEQUAL', 46: 'DOUBLESTAREQUAL', 47: 'DOUBLESLASH', 48: 'DOUBLESLASHEQUAL', 49: 'AT', 50: 'ATEQUAL', 51: 'RARROW', 52: 'ELLIPSIS', 53: 'COLONEQUAL', 54: 'OP', 55: 'AWAIT', 56: 'ASYNC', 57: 'TYPE_IGNORE', 58: 'TYPE_COMMENT', 59: 'SOFT_KEYWORD', 60: 'ERRORTOKEN', 61: 'COMMENT', 62: 'NL', 63: 'ENCODING', 64: 'N_TOKENS', 256: 'NT_OFFSET'}
    assert module_5.ENDMARKER == 0
    assert module_5.NAME == 1
    assert module_5.NUMBER == 2
    assert module_5.STRING == 3
    assert module_5.NEWLINE == 4
    assert module_5.INDENT == 5
    assert module_5.DEDENT == 6
    assert module_5.LPAR == 7
    assert module_5.RPAR == 8
    assert module_5.LSQB == 9
    assert module_5.RSQB == 10
    assert module_5.COLON == 11
    assert module_5.COMMA == 12
    assert module_5.SEMI == 13
    assert module_5.PLUS == 14
    assert module_5.MINUS == 15
    assert module_5.STAR == 16
    assert module_5.SLASH == 17
    assert module_5.VBAR == 18
    assert module_5.AMPER == 19
    assert module_5.LESS == 20
    assert module_5.GREATER == 21
    assert module_5.EQUAL == 22
    assert module_5.DOT == 23
    assert module_5.PERCENT == 24
    assert module_5.LBRACE == 25
    assert module_5.RBRACE == 26
    assert module_5.EQEQUAL == 27
    assert module_5.NOTEQUAL == 28
    assert module_5.LESSEQUAL == 29
    assert module_5.GREATEREQUAL == 30
    assert module_5.TILDE == 31
    assert module_5.CIRCUMFLEX == 32
    assert module_5.LEFTSHIFT == 33
    assert module_5.RIGHTSHIFT == 34
    assert module_5.DOUBLESTAR == 35
    assert module_5.PLUSEQUAL == 36
    assert module_5.MINEQUAL == 37
    assert module_5.STAREQUAL == 38
    assert module_5.SLASHEQUAL == 39
    assert module_5.PERCENTEQUAL == 40
    assert module_5.AMPEREQUAL == 41
    assert module_5.VBAREQUAL == 42
    assert module_5.CIRCUMFLEXEQUAL == 43
    assert module_5.LEFTSHIFTEQUAL == 44
    assert module_5.RIGHTSHIFTEQUAL == 45
    assert module_5.DOUBLESTAREQUAL == 46
    assert module_5.DOUBLESLASH == 47
    assert module_5.DOUBLESLASHEQUAL == 48
    assert module_5.AT == 49
    assert module_5.ATEQUAL == 50
    assert module_5.RARROW == 51
    assert module_5.ELLIPSIS == 52
    assert module_5.COLONEQUAL == 53
    assert module_5.OP == 54
    assert module_5.AWAIT == 55
    assert module_5.ASYNC == 56
    assert module_5.TYPE_IGNORE == 57
    assert module_5.TYPE_COMMENT == 58
    assert module_5.SOFT_KEYWORD == 59
    assert module_5.ERRORTOKEN == 60
    assert module_5.COMMENT == 61
    assert module_5.NL == 62
    assert module_5.ENCODING == 63
    assert module_5.N_TOKENS == 64
    assert module_5.NT_OFFSET == 256
    assert module_5.EXACT_TOKEN_TYPES == {'!=': 28, '%': 24, '%=': 40, '&': 19, '&=': 41, '(': 7, ')': 8, '*': 16, '**': 35, '**=': 46, '*=': 38, '+': 14, '+=': 36, ',': 12, '-': 15, '-=': 37, '->': 51, '.': 23, '...': 52, '/': 17, '//': 47, '//=': 48, '/=': 39, ':': 11, ':=': 53, ';': 13, '<': 20, '<<': 33, '<<=': 44, '<=': 29, '=': 22, '==': 27, '>': 21, '>=': 30, '>>': 34, '>>=': 45, '@': 49, '@=': 50, '[': 9, ']': 10, '^': 32, '^=': 43, '{': 25, '|': 18, '|=': 42, '}': 26, '~': 31}
    assert f'{type(module_5.cookie_re).__module__}.{type(module_5.cookie_re).__qualname__}' == 're.Pattern'
    assert f'{type(module_5.blank_re).__module__}.{type(module_5.blank_re).__qualname__}' == 're.Pattern'
    assert module_5.Whitespace == '[ \\f\\t]*'
    assert module_5.Comment == '#[^\\r\\n]*'
    assert module_5.Ignore == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?'
    assert module_5.Name == '\\w+'
    assert module_5.Hexnumber == '0[xX](?:_?[0-9a-fA-F])+'
    assert module_5.Binnumber == '0[bB](?:_?[01])+'
    assert module_5.Octnumber == '0[oO](?:_?[0-7])+'
    assert module_5.Decnumber == '(?:0(?:_?0)*|[1-9](?:_?[0-9])*)'
    assert module_5.Intnumber == '(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*))'
    assert module_5.Exponent == '[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_5.Pointfloat == '([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?'
    assert module_5.Expfloat == '[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*'
    assert module_5.Floatnumber == '(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)'
    assert module_5.Imagnumber == '([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])'
    assert module_5.Number == '(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))'
    assert module_5.StringPrefix == '(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)'
    assert module_5.Single == "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'"
    assert module_5.Double == '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"'
    assert module_5.Single3 == "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''"
    assert module_5.Double3 == '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'
    assert module_5.Triple == '((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'\'\'|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)""")'
    assert module_5.String == '((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")'
    assert module_5.Special == '(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=)'
    assert module_5.Funny == '(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))'
    assert module_5.PlainToken == '((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_5.Token == '[ \\f\\t]*(\\\\\\r?\\n[ \\f\\t]*)*(#[^\\r\\n]*)?((([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*\'|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*")|\\w+)'
    assert module_5.ContStr == '((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))'
    assert module_5.PseudoExtras == '(\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'\'\'|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"""))'
    assert module_5.PseudoToken == '[ \\f\\t]*((\\\\\\r?\\n|\\Z|#[^\\r\\n]*|((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'\'\'|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"""))|(([0-9](?:_?[0-9])*[jJ]|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)[jJ])|(([0-9](?:_?[0-9])*\\.(?:[0-9](?:_?[0-9])*)?|\\.[0-9](?:_?[0-9])*)([eE][-+]?[0-9](?:_?[0-9])*)?|[0-9](?:_?[0-9])*[eE][-+]?[0-9](?:_?[0-9])*)|(0[xX](?:_?[0-9a-fA-F])+|0[bB](?:_?[01])+|0[oO](?:_?[0-7])+|(?:0(?:_?0)*|[1-9](?:_?[0-9])*)))|(\\r?\\n|(\\~|\\}|\\|=|\\||\\{|\\^=|\\^|\\]|\\[|@=|@|>>=|>>|>=|>|==|=|<=|<<=|<<|<|;|:=|:|/=|//=|//|/|\\.\\.\\.|\\.|\\->|\\-=|\\-|,|\\+=|\\+|\\*=|\\*\\*=|\\*\\*|\\*|\\)|\\(|\\&=|\\&|%=|%|!=))|((|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)\'[^\\n\'\\\\]*(?:\\\\.[^\\n\'\\\\]*)*(\'|\\\\\\r?\\n)|(|br|U|rb|rf|u|RF|f|r|Br|bR|RB|FR|B|R|Rf|BR|Rb|F|rF|fR|b|Fr|rB|fr)"[^\\n"\\\\]*(?:\\\\.[^\\n"\\\\]*)*("|\\\\\\r?\\n))|\\w+)'
    assert module_5.endpats == {"'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", '"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", '"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "U'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'U"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "U'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'U"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "u'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'u"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "u'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'u"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "f'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'f"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "f'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'f"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "r'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'r"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "r'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'r"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Br'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Br"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Br'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Br"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "bR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'bR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "bR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'bR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "RB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'RB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "RB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'RB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "FR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'FR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "FR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'FR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "B'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'B"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "B'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'B"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "R'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'R"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "R'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'R"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rf'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rf"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rf'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rf"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "BR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'BR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "BR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'BR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Rb'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Rb"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Rb'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Rb"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "F'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'F"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "F'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'F"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rF'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rF"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rF'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rF"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fR'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fR"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fR'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fR"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "b'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'b"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "b'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'b"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "Fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'Fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "Fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'Fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "rB'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'rB"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "rB'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'rB"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""', "fr'": "[^'\\\\]*(?:\\\\.[^'\\\\]*)*'", 'fr"': '[^"\\\\]*(?:\\\\.[^"\\\\]*)*"', "fr'''": "[^'\\\\]*(?:(?:\\\\.|'(?!''))[^'\\\\]*)*'''", 'fr"""': '[^"\\\\]*(?:(?:\\\\.|"(?!""))[^"\\\\]*)*"""'}
    assert module_5.single_quoted == {'rF"', 'RB"', "U'", "BR'", 'BR"', 'RF"', "fr'", 'u"', 'U"', 'Br"', 'R"', "rF'", "r'", "R'", "rb'", 'rb"', 'FR"', 'Rf"', 'fr"', 'rf"', "Rb'", "F'", "b'", "RB'", "FR'", "Fr'", "rf'", 'rB"', "bR'", "Rf'", 'B"', "B'", 'Rb"', "'", "rB'", 'f"', 'r"', "u'", "br'", 'br"', "Br'", "fR'", "f'", 'fR"', 'bR"', 'Fr"', "RF'", 'F"', '"', 'b"'}
    assert module_5.triple_quoted == {"fr'''", "Rb'''", "RB'''", "bR'''", 'FR"""', 'Fr"""', "'''", 'Rb"""', 'f"""', 'F"""', 'rB"""', "rf'''", 'rb"""', 'Rf"""', "fR'''", 'fr"""', 'rF"""', "rB'''", 'u"""', 'U"""', "br'''", 'fR"""', 'br"""', "f'''", "b'''", "RF'''", "Fr'''", "F'''", "U'''", 'B"""', 'RB"""', "R'''", 'RF"""', 'b"""', "FR'''", "BR'''", 'r"""', "r'''", 'BR"""', 'Br"""', 'R"""', "u'''", "Br'''", '"""', "Rf'''", 'rf"""', "rF'''", 'bR"""', "rb'''", "B'''"}
    assert module_5.t == 'fr'
    assert module_5.u == "fr'''"
    assert module_5.tabsize == 8
    var_2 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_1)
    assert var_2 is False
    var_3 = module_0.check_type(var_0, var_0, var_0, var_0)
    var_4 = module_1.CheckedPVector()
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_4) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_5 = var_4.evolver()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector.Evolver'
    assert len(var_5) == 0
    var_6 = module_0.pmap_field(var_4, var_4, var_4)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._field_common._PField'
    var_7 = module_3.Slice()
    assert f'{type(var_7).__module__}.{type(var_7).__qualname__}' == 'ast.Slice'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.Slice.lower is None
    assert module_3.Slice.upper is None
    assert module_3.Slice.step is None
    module_0.check_type(var_3, var_6, var_3, var_6)

@pytest.mark.xfail(strict=True)
def test_case_27():
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
    var_1 = module_0.field(initial=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    var_2 = module_0.check_type(var_0, var_0, var_0, var_0)
    var_3 = module_1.CheckedPVector()
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._checked_types.CheckedPVector'
    assert len(var_3) == 0
    assert f'{type(module_1.T_co).__module__}.{type(module_1.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.KT).__module__}.{type(module_1.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.VT_co).__module__}.{type(module_1.VT_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_1.CheckedPVector.create).__module__}.{type(module_1.CheckedPVector.create).__qualname__}' == 'builtins.method'
    var_4 = module_0.pmap_field(var_3, var_3, var_1)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._field_common._PField'
    var_5 = module_3.Slice()
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'ast.Slice'
    assert module_3.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_3.PyCF_ONLY_AST == 1024
    assert module_3.PyCF_TYPE_COMMENTS == 4096
    assert module_3.Slice.lower is None
    assert module_3.Slice.upper is None
    assert module_3.Slice.step is None
    var_6 = module_0.is_field_ignore_extra_complaint(var_4, var_0, var_0)
    assert var_6 is False
    var_7 = None
    var_8 = module_0.is_field_ignore_extra_complaint(var_7, var_0, var_0)
    assert var_8 is False
    var_9 = module_0.check_type(var_6, var_4, var_7, var_7)
    module_0.serialize(var_4, var_1, var_3)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = module_3.Param
    var_1 = None
    module_0.pvector_field(var_0, initial=var_1)