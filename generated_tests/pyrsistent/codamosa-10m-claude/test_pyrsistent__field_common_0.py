# Check out: https://github.com/GlowCheese/deepmosa
import ast as module_1
import enum as module_2

import pyrsistent._checked_types as module_3
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
    var_0 = module_1.Or
    var_1 = module_0.pvector_field(var_0, item_invariant=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    module_0.check_type(var_1, var_1, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_5():
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
    module_0.field(invariant=var_0)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = 'ae'
    module_0.field(var_0, var_0, var_0, var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    module_0.field(serializer=var_0)

@pytest.mark.xfail(strict=True)
def test_case_8():
    var_0 = None
    var_1 = module_0.field(initial=var_0, mandatory=var_0)
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
    module_0.pmap_field(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_9():
    var_0 = None
    module_0.field(initial=var_0, mandatory=var_0, factory=var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
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
    module_0.pset_field(var_0, initial=var_0, invariant=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = None
    module_0.pmap_field(var_0, var_0)

def test_case_12():
    var_0 = None
    var_1 = []
    var_2 = module_0.PTypeError(var_0, var_0, var_0, var_0, *var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common.PTypeError'
    assert var_2.source_class is None
    assert var_2.field is None
    assert var_2.expected_types is None
    assert var_2.actual_type is None
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

@pytest.mark.xfail(strict=True)
def test_case_13():
    var_0 = None
    module_0.pvector_field(var_0)

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

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = module_2._EnumDict()
    assert f'{type(var_0).__module__}.{type(var_0).__qualname__}' == 'enum._EnumDict'
    assert len(var_0) == 0
    var_1 = module_0.is_type_cls(var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_2 = None
    module_0.pmap_field(var_1, var_2)

def test_case_16():
    var_0 = module_1.Or
    var_1 = module_0.pvector_field(var_0, item_invariant=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    var_2 = module_0.field()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    var_3 = module_0.check_type(var_2, var_2, var_2, var_2)
    var_4 = None
    var_5 = None
    var_6 = module_0._PField(var_5, var_3, var_3, var_4, var_1, var_2)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._field_common._PField'

@pytest.mark.xfail(strict=True)
def test_case_17():
    var_0 = module_1.Or
    var_1 = module_0.pvector_field(var_0, item_invariant=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    var_2 = module_0.field()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    var_3 = None
    var_4 = module_0.check_type(var_2, var_2, var_3, var_3)
    var_5 = module_0.is_field_ignore_extra_complaint(var_3, var_1, var_2)
    assert var_5 is True
    var_6 = module_0.is_field_ignore_extra_complaint(var_2, var_3, var_3)
    assert var_6 is False
    var_4.mset()

@pytest.mark.xfail(strict=True)
def test_case_18():
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
    var_2 = var_0.__repr__()
    var_3 = module_0.is_field_ignore_extra_complaint(var_1, var_0, var_2)
    assert var_3 is False
    module_0.is_type_cls(var_0, var_2)

def test_case_19():
    var_0 = module_1.Or
    var_1 = module_0.field(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    var_2 = module_0.field(var_0, var_0, mandatory=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'

def test_case_20():
    var_0 = {}
    var_1 = None
    var_2 = module_0.check_global_invariants(var_1, var_0)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

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
    var_1 = None
    var_2 = module_0.check_type(var_0, var_0, var_1, var_1)
    var_3 = var_2.__repr__()
    assert var_3 == 'None'
    module_0.check_global_invariants(var_2, var_3)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = module_1.And
    var_1 = None
    module_0.field(initial=var_0, factory=var_1)

@pytest.mark.xfail(strict=True)
def test_case_23():
    var_0 = module_1.Or
    var_1 = module_0.pvector_field(var_0, var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    var_2 = module_0.field(invariant=var_0, mandatory=var_1, serializer=var_0)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    var_3 = None
    module_0.check_type(var_2, var_3, var_1, var_1)

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = module_1.And
    var_1 = module_0.pvector_field(var_0, item_invariant=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    module_0.field(var_0, initial=var_1, mandatory=var_1, factory=var_1)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = module_1.Or
    var_1 = module_0.field(invariant=var_0)
    assert f'{type(var_1).__module__}.{type(var_1).__qualname__}' == 'pyrsistent._field_common._PField'
    assert module_1.PyCF_ALLOW_TOP_LEVEL_AWAIT == 8192
    assert module_1.PyCF_ONLY_AST == 1024
    assert module_1.PyCF_TYPE_COMMENTS == 4096
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
    module_0.pvector_field(var_0, var_1, var_2, var_1)

def test_case_26():
    var_0 = {}
    var_1 = 'HC^%WXPP8+=\x0b|WgzCG'
    var_2 = set()
    var_3 = module_0.set_fields(var_0, var_2, var_1)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

@pytest.mark.xfail(strict=True)
def test_case_27():
    var_0 = None
    var_1 = module_3.optional()
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = '3Kw.KDt)s2}'
    var_3 = 'value2'
    var_4 = {}
    var_5 = module_0.pmap_field(var_1, var_1)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._field_common._PField'
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
    var_6 = module_0._PField(var_1, var_3, var_0, var_3, var_3, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._field_common._PField'
    var_7 = module_0.check_global_invariants(var_4, var_4)
    var_8 = set()
    var_9 = '5'
    var_10 = module_0.field(initial=var_0)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._field_common._PField'
    module_0.set_fields(var_4, var_2, var_9)

@pytest.mark.xfail(strict=True)
def test_case_28():
    var_0 = 'ield1'
    var_1 = None
    var_2 = module_0._PField(var_0, var_1, var_1, var_1, var_1, var_1)
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
    var_3 = module_0._PField(var_1, var_1, var_1, var_1, var_1, var_1)
    assert f'{type(var_3).__module__}.{type(var_3).__qualname__}' == 'pyrsistent._field_common._PField'
    var_4 = 'value2'
    var_5 = {var_3: var_2}
    var_6 = set()
    var_7 = module_0.check_global_invariants(var_6, var_6)
    var_8 = set()
    var_9 = 'pf1'
    var_10 = module_0.field(initial=var_6)
    assert f'{type(var_10).__module__}.{type(var_10).__qualname__}' == 'pyrsistent._field_common._PField'
    var_11 = module_0.set_fields(var_5, var_6, var_9)
    var_12 = module_0.is_field_ignore_extra_complaint(var_4, var_6, var_11)
    assert var_12 is False
    var_13 = var_12.__dir__()
    module_0.field(initial=var_13, serializer=var_1)

def test_case_29():
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
    var_1 = module_3.optional()
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_2 = module_0.pmap_field(var_1, var_1)
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    var_3 = set()
    var_4 = module_0.check_global_invariants(var_3, var_3)
    var_5 = 'inherited_value'
    var_6 = module_0.PTypeError(var_1, var_5, var_1, var_1)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._field_common.PTypeError'
    assert f'{type(var_6.source_class).__module__}.{type(var_6.source_class).__qualname__}' == 'builtins.tuple'
    assert len(var_6.source_class) == 1
    assert var_6.field == 'inherited_value'
    assert f'{type(var_6.expected_types).__module__}.{type(var_6.expected_types).__qualname__}' == 'builtins.tuple'
    assert len(var_6.expected_types) == 1
    assert f'{type(var_6.actual_type).__module__}.{type(var_6.actual_type).__qualname__}' == 'builtins.tuple'
    assert len(var_6.actual_type) == 1

@pytest.mark.xfail(strict=True)
def test_case_30():
    var_0 = None
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_2 = module_0.field()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_3.optional()
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = module_0.pmap_field(var_3, var_3, var_3)
    assert f'{type(var_4).__module__}.{type(var_4).__qualname__}' == 'pyrsistent._field_common._PField'
    var_5 = module_0.pmap_field(var_3, var_3)
    assert f'{type(var_5).__module__}.{type(var_5).__qualname__}' == 'pyrsistent._field_common._PField'
    var_6 = set()
    var_2.__new__(var_5, var_5)

@pytest.mark.xfail(strict=True)
def test_case_31():
    var_0 = None
    var_1 = module_0.is_field_ignore_extra_complaint(var_0, var_0, var_0)
    assert var_1 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_2 = module_0.field()
    assert f'{type(var_2).__module__}.{type(var_2).__qualname__}' == 'pyrsistent._field_common._PField'
    assert f'{type(module_0._PField.factory).__module__}.{type(module_0._PField.factory).__qualname__}' == 'builtins.property'
    assert f'{type(module_0._PField.initial).__module__}.{type(module_0._PField.initial).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.invariant).__module__}.{type(module_0._PField.invariant).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.mandatory).__module__}.{type(module_0._PField.mandatory).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.serializer).__module__}.{type(module_0._PField.serializer).__qualname__}' == 'builtins.member_descriptor'
    assert f'{type(module_0._PField.type).__module__}.{type(module_0._PField.type).__qualname__}' == 'builtins.member_descriptor'
    var_3 = module_3.optional()
    assert f'{type(module_3.T_co).__module__}.{type(module_3.T_co).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.KT).__module__}.{type(module_3.KT).__qualname__}' == 'typing.TypeVar'
    assert f'{type(module_3.VT_co).__module__}.{type(module_3.VT_co).__qualname__}' == 'typing.TypeVar'
    var_4 = set()
    var_5 = module_0.check_global_invariants(var_4, var_4)
    var_6 = module_0.field(var_3, mandatory=var_5)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._field_common._PField'
    var_7 = module_0.check_type(var_0, var_6, var_4, var_0)
    module_0.serialize(var_0, var_4, var_2)