# Check out: https://github.com/GlowCheese/deepmosa
import collections as module_1

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

def test_case_4():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    module_0.pvector_field(var_1, var_1, item_invariant=var_0)
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

@pytest.mark.xfail(strict=True)
def test_case_5():
    var_0 = 'I2V,\t8(lzl@:WW3'
    var_1 = None
    module_0.field(var_0, var_1, serializer=var_1)

@pytest.mark.xfail(strict=True)
def test_case_6():
    var_0 = None
    module_0.pvector_field(var_0)

@pytest.mark.xfail(strict=True)
def test_case_7():
    var_0 = None
    module_0.pmap_field(var_0, var_0)

def test_case_8():
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
def test_case_9():
    var_0 = None
    module_0.pset_field(var_0, var_0, var_0)

@pytest.mark.xfail(strict=True)
def test_case_10():
    var_0 = None
    module_0.field(initial=var_0, serializer=var_0)

@pytest.mark.xfail(strict=True)
def test_case_11():
    var_0 = []
    var_1 = None
    var_2 = module_0.is_field_ignore_extra_complaint(var_1, var_1, var_1)
    assert var_2 is False
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_3 = module_0.check_global_invariants(var_1, var_0)
    module_0.field(factory=var_2)

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
    var_1 = module_0.check_type(var_0, var_0, var_0, var_0)

def test_case_14():
    var_0 = []
    var_1 = module_0.check_global_invariants(var_0, var_0)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2

@pytest.mark.xfail(strict=True)
def test_case_15():
    var_0 = 'test'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    module_0.check_global_invariants(var_0, var_7)

def test_case_16():
    var_0 = {}
    var_1 = ()
    var_2 = 'test_field'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_4 = bool(var_0[var_2] == {})
    assert var_4 is True

def test_case_17():
    var_0 = 'builtins.int'
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

def test_case_18():
    var_0 = 'default'
    var_1 = lambda x: x
    var_2 = module_0.field(invariant=var_1, initial=var_0, factory=var_1, serializer=var_1)
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
    var_3 = module_0._check_field_parameters(var_2)

@pytest.mark.xfail(strict=True)
def test_case_19():
    var_0 = None
    var_1 = 'Seq'
    module_0.is_type_cls(var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_20():
    var_0 = []
    var_1 = False
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 10
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 15
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0.field(initial=var_8)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._field_common._PField'
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
    module_0.check_global_invariants(var_0, var_12)

@pytest.mark.xfail(strict=True)
def test_case_21():
    var_0 = 'tes_field'
    var_1 = None
    module_0.set_fields(var_1, var_0, var_1)

@pytest.mark.xfail(strict=True)
def test_case_22():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = None
    var_3 = module_0.pvector_field(var_1)
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
    module_0.check_type(var_3, var_3, var_2, var_0)

def test_case_23():
    var_0 = 'initial'
    var_1 = 'type'
    var_2 = [var_0, var_1]
    var_3 = module_1.namedtuple(var_0, var_2)
    var_4 = []
    var_5 = module_0.pmap_field(var_3, var_3, var_3)
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
    var_6 = module_0.pmap_field(var_3, var_3)
    assert f'{type(var_6).__module__}.{type(var_6).__qualname__}' == 'pyrsistent._field_common._PField'
    with pytest.raises(AttributeError):
        var_7 = var_4.initial

@pytest.mark.xfail(strict=True)
def test_case_24():
    var_0 = None
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)
    assert module_0.PFIELD_NO_TYPE == ()
    assert f'{type(module_0.PFIELD_NO_INITIAL).__module__}.{type(module_0.PFIELD_NO_INITIAL).__qualname__}' == 'builtins.object'
    assert f'{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__module__}.{type(module_0.SEQ_FIELD_TYPE_SUFFIXES).__qualname__}' == 'builtins.dict'
    assert len(module_0.SEQ_FIELD_TYPE_SUFFIXES) == 2
    var_3 = module_0.is_type_cls(var_2, var_1)
    assert var_3 is False
    module_0.is_type_cls(var_2, var_2)

@pytest.mark.xfail(strict=True)
def test_case_25():
    var_0 = 'initial'
    module_0.field(var_0)

@pytest.mark.xfail(strict=True)
def test_case_26():
    var_0 = 'TestClass'
    var_1 = 'TestItem'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = lambda self, data, _factory_fields: data
    module_0._restore_seq_field_pickle(var_0, var_1, var_5)

def test_case_27():
    var_0 = 'initial'
    var_1 = 'type'
    var_2 = None
    var_3 = module_0.field(initial=var_2)
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
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = None
    var_7 = module_0.is_field_ignore_extra_complaint(var_6, var_6, var_6)
    assert var_7 is False
    var_8 = module_0.is_field_ignore_extra_complaint(var_6, var_3, var_3)
    assert var_8 is False
    var_9 = 'serializer'
    var_10 = [var_0, var_1, var_4, var_5, var_9]
    var_11 = module_1.namedtuple(var_9, var_10)
    var_12 = None
    var_13 = module_0.pset_field(var_11, var_11, item_invariant=var_11)
    assert f'{type(var_13).__module__}.{type(var_13).__qualname__}' == 'pyrsistent._field_common._PField'
    var_14 = module_0.check_type(var_11, var_13, var_13, var_12)
    var_15 = var_10.__repr__()
    assert var_15 == "['initial', 'type', 'invariant', 'factory', 'serializer']"
    var_16 = 123
    var_17 = lambda : var_16
    var_18 = True
    var_19 = lambda x: var_18
    var_20 = None
    var_21 = lambda : var_20
    var_22 = lambda x: x