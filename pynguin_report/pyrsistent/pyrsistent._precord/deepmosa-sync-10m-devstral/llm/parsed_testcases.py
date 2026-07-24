####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_with_invariant_errors. Retrieved 11/20 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 11/20 statements.


import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = module_2._PRecordEvolver(var_12, var_15)
    var_17 = 'key'
    var_18 = 'value'
    var_19 = var_16.set(var_17, var_18)
    var_20 = var_16.persistent()
    var_21 = isinstance(var_20, var_12)
    var_22 = bool(var_21)
    assert var_22 is True
    var_23 = var_20['key']
    assert var_23 == 'value'

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = var_15._buckets
    var_17 = []
    var_18 = {}
    var_19 = module_1.PMap(*var_17, **var_18)
    var_20 = var_19._size
    var_21 = var_12(_precord_buckets=var_16, _precord_size=var_20)
    var_22 = module_2._PRecordEvolver(var_12, var_21)
    var_23 = var_22.persistent()
    var_24 = bool(var_23 is var_21)
    assert var_24 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = 'mandatory_field'
    var_8 = {var_7}
    var_9 = []
    var_10 = {var_2: var_6, var_3: var_8, var_4: var_9, var_5: var_0}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = []
    var_15 = {}
    var_16 = module_1.PMap(*var_14, **var_15)
    var_17 = module_2._PRecordEvolver(var_13, var_16)
    var_18 = var_17.persistent()
    var_19 = bool(False)
    assert var_19 is True
    var_20 = 'MockClass.mandatory_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'INVARIANT_FAILED'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Global invariant failed'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__new__with_no_bases_and_empty_dct. Retrieved 4/5 statements.
# Partially parsed test__new__with_single_base_and_field_in_dct. Retrieved 3/9 statements.
# Partially parsed test__new__with_inherited_fields. Retrieved 2/6 statements.
# Partially parsed test__new__with_inherited_invariants. Retrieved 4/11 statements.
# Partially parsed test__new__with_multiple_inherited_invariants. Retrieved 4/13 statements.
# Partially parsed test__new__with_non_callable_invariant_raises_type_error. Retrieved 3/7 statements.
# Partially parsed test__new__with_field_initial_value. Retrieved 4/7 statements.
# Partially parsed test__new__with_no_initial_value_field. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = set()

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 'TestClass'
    var_3 = 'field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 'TestClass'
    var_3 = {}
    var_4 = 'field1'

def test_case_0():
    var_0 = lambda self: True
    var_1 = 'TestClass'
    var_2 = {}
    var_3 = 0

def test_case_0():
    var_0 = lambda self: True
    var_1 = lambda self: (True, 'test')
    var_2 = 'TestClass'
    var_3 = {}

def test_case_0():
    var_0 = 'not callable'
    var_1 = 'TestClass'
    var_2 = {}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 42
    var_2 = 'TestClass'
    var_3 = ()

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestClass'
    var_2 = ()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 4/8 statements.
# Partially parsed test_set_with_invalid_type. Retrieved 4/9 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 4/8 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 5/9 statements.
# Partially parsed test_set_with_ignore_extra_compliant_field. Retrieved 5/9 statements.
# Partially parsed test_set_with_factory_field_not_in_factory_fields. Retrieved 5/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = 'field1'
    var_6 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = 'field1'
    var_6 = 'not_an_int'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = 'field1'
    var_6 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = 'nonexistent'
    var_6 = 10
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = True
    var_6 = 'field1'
    var_7 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = set()
    var_6 = 'field1'
    var_7 = 10



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 2/7 statements.
# Failed to parse test_precord_new_without_special_attributes.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/3 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'TestRecord'
    var_1 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = {var_0}
    var_2 = 1
    var_3 = 2

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = 'c'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_with_invariant_errors. Retrieved 15/17 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 8/16 statements.


import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = []
    var_13 = {}
    var_14 = module_1.PMap(*var_12, **var_13)
    var_15 = module_2._PRecordEvolver(var_11, var_14)
    var_16 = 'key'
    var_17 = 'value'
    var_18 = var_15.set(var_16, var_17)
    var_19 = var_15.persistent()
    var_20 = isinstance(var_19, var_11)
    var_21 = bool(var_20)
    assert var_21 is True
    var_22 = var_19._precord_buckets
    var_23 = bool(var_19._precord_buckets == var_15._buckets)
    assert var_23 is True
    var_24 = var_19._precord_size
    var_25 = bool(var_19._precord_size == var_15._size)
    assert var_25 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = {}
    var_13 = 0
    var_14 = var_11(_precord_buckets=var_12, _precord_size=var_13)
    var_15 = module_1._PRecordEvolver(var_11, var_14)
    var_16 = var_15.persistent()
    var_17 = bool(var_16 is var_14)
    assert var_17 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = module_2._PRecordEvolver(var_12, var_15)
    var_17 = var_16.persistent()
    var_18 = bool(False)
    assert var_18 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = []
    var_13 = {}
    var_14 = module_1.PMap(*var_12, **var_13)
    var_15 = module_2._PRecordEvolver(var_11, var_14)
    var_16 = 'error1'
    var_17 = 'error2'
    var_18 = var_15.persistent()
    var_19 = bool(False)
    assert var_19 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_persistent_with_dirty_state_and_non_instance. Retrieved 24/25 statements.
# Partially parsed test_persistent_with_clean_state_and_instance. Retrieved 28/30 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 32/35 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 29/33 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 23/33 statements.


import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = 'MockPMap'
    var_14 = ()
    var_15 = '_buckets'
    var_16 = '_size'
    var_17 = {}
    var_18 = 0
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_13, var_14, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1._PRecordEvolver(var_12, var_23)
    var_25 = True
    var_26 = var_24.persistent()
    var_27 = isinstance(var_26, var_12)
    var_28 = bool(var_27)
    assert var_28 is True
    var_29 = var_26._precord_buckets
    var_30 = bool(var_26._precord_buckets == var_23._buckets)
    assert var_30 is True
    var_31 = var_26._precord_size
    var_32 = bool(var_26._precord_size == var_23._size)
    assert var_32 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = 'MockPMap'
    var_14 = ()
    var_15 = '_buckets'
    var_16 = '_size'
    var_17 = {}
    var_18 = 0
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_13, var_14, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1._PRecordEvolver(var_12, var_23)
    var_25 = False
    var_26 = ()
    var_27 = {}
    var_28 = {var_15: var_27, var_16: var_25}
    var_29 = [var_13, var_26, var_28]
    var_30 = {}
    var_31 = module_0.type(*var_29, **var_30)
    var_32 = var_31()
    var_33 = var_24.persistent()
    var_34 = bool(var_33 is var_32)
    assert var_34 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = 'field1'
    var_8 = {var_7}
    var_9 = []
    var_10 = {var_2: var_6, var_3: var_8, var_4: var_9, var_5: var_0}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = 'MockPMap'
    var_15 = ()
    var_16 = '_buckets'
    var_17 = '_size'
    var_18 = {}
    var_19 = 0
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = [var_14, var_15, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1._PRecordEvolver(var_13, var_24)
    var_26 = False
    var_27 = ()
    var_28 = 'keys'
    var_29 = {}
    var_30 = []
    var_31 = lambda : var_30
    var_32 = {var_16: var_29, var_17: var_26, var_28: var_31}
    var_33 = [var_14, var_27, var_32]
    var_34 = {}
    var_35 = module_0.type(*var_33, **var_34)
    var_36 = var_35()
    var_37 = var_25.persistent()
    var_38 = bool(False)
    assert var_38 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = 'MockPMap'
    var_14 = ()
    var_15 = '_buckets'
    var_16 = '_size'
    var_17 = {}
    var_18 = 0
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_13, var_14, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1._PRecordEvolver(var_12, var_23)
    var_25 = 'error1'
    var_26 = False
    var_27 = ()
    var_28 = {}
    var_29 = {var_15: var_28, var_16: var_26}
    var_30 = [var_13, var_27, var_29]
    var_31 = {}
    var_32 = module_0.type(*var_30, **var_31)
    var_33 = var_32()
    var_34 = var_24.persistent()
    var_35 = bool(False)
    assert var_35 is True

import builtins as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = 'MockPMap'
    var_9 = ()
    var_10 = '_buckets'
    var_11 = '_size'
    var_12 = {}
    var_13 = 0
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = [var_8, var_9, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = False
    var_20 = ()
    var_21 = {}
    var_22 = {var_10: var_21, var_11: var_19}
    var_23 = [var_8, var_20, var_22]
    var_24 = {}
    var_25 = module_0.type(*var_23, **var_24)
    var_26 = var_25()
    var_27 = bool(False)
    assert var_27 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_no_custom_serializers. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_custom_serializers. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 5/8 statements.
# Failed to parse test_serialize_empty_record.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = module_0.field(serializer=var_0)
    var_2 = lambda x: x.upper()
    var_3 = module_0.field(serializer=var_2)
    var_4 = 10
    var_5 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x, fmt: f'{x}:{fmt}'
    var_1 = module_0.field(serializer=var_0)
    var_2 = lambda x, fmt: f'{x.upper()}-{fmt}'
    var_3 = module_0.field(serializer=var_2)
    var_4 = 10
    var_5 = 'test'
    var_6 = 'json'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_repr_empty.
# Partially parsed test_repr_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 4/7 statements.
# Partially parsed test_repr_with_complex_values. Retrieved 9/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_invariant_exception_raised_when_error_codes_or_missing_fields. Retrieved 6/12 statements.
# Partially parsed test_invariant_exception_raised_when_missing_fields. Retrieved 6/12 statements.
# Partially parsed test_invariant_exception_raised_when_both_error_codes_and_missing_fields. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'error1'
    var_1 = 'MockClass'
    var_2 = ()
    var_3 = '_precord_invariants'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'MockClass'
    var_2 = ()
    var_3 = '_precord_invariants'
    var_4 = []
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = {}

def test_case_0():
    var_0 = 'error1'
    var_1 = 'field1'
    var_2 = 'MockClass'
    var_3 = ()
    var_4 = '_precord_invariants'
    var_5 = []
    var_6 = {var_4: var_5}
    var_7 = [var_2, var_3, var_6]
    var_8 = {}



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_repr_empty_record.
# Partially parsed test_repr_simple_record. Retrieved 4/7 statements.
# Partially parsed test_repr_nested_record. Retrieved 5/9 statements.
# Partially parsed test_repr_with_special_characters. Retrieved 4/7 statements.
# Partially parsed test_repr_with_none_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'
    var_4 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = "test's record"
    var_3 = '/path/to/file'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 100



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 5/11 statements.
# Partially parsed test_set_with_invalid_field_type. Retrieved 5/12 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 5/11 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 4/8 statements.
# Partially parsed test_set_with_ignore_extra_and_compliant_factory. Retrieved 6/12 statements.
# Partially parsed test_set_with_factory_fields_restriction. Retrieved 6/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value: value * 2
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'field1'
    var_7 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value: value
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'field1'
    var_7 = 'not_an_int'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (False, 'INVALID') if value < 0 else (True, None)
    var_1 = lambda value: value
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'field1'
    var_7 = -5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = {}
    var_3 = module_0.PMap(*var_1, **var_2)
    var_4 = 'nonexistent_field'
    var_5 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value, ignore_extra=False: value if ignore_extra else value * 2
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = True
    var_7 = 'field1'
    var_8 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value: value * 2
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'field1'
    var_8 = 5
    var_9 = 'field2'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__new__sets_fields_and_invariants. Retrieved 1/8 statements.


def test_case_0():
    var_0 = lambda self: (True, 'test')
    var_1 = '_precord_fields'
    var_2 = 'x'
    var_3 = 'y'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 18/19 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 11/20 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 9/17 statements.


import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = module_2._PRecordEvolver(var_12, var_15)
    var_17 = 'key'
    var_18 = 'value'
    var_19 = var_16.set(var_17, var_18)
    var_20 = var_16.persistent()
    var_21 = isinstance(var_20, var_12)
    var_22 = bool(var_21)
    assert var_22 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = var_15._buckets
    var_17 = []
    var_18 = {}
    var_19 = module_1.PMap(*var_17, **var_18)
    var_20 = var_19._size
    var_21 = var_12(_precord_buckets=var_16, _precord_size=var_20)
    var_22 = module_2._PRecordEvolver(var_12, var_21)
    var_23 = var_22.persistent()
    var_24 = bool(var_23 is var_21)
    assert var_24 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = 'mandatory_field'
    var_8 = {var_7}
    var_9 = []
    var_10 = {var_2: var_6, var_3: var_8, var_4: var_9, var_5: var_0}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = []
    var_15 = {}
    var_16 = module_1.PMap(*var_14, **var_15)
    var_17 = module_2._PRecordEvolver(var_13, var_16)
    var_18 = var_17.persistent()
    var_19 = bool(False)
    assert var_19 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = 'key'
    var_12 = 'value'
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_instance. Retrieved 24/25 statements.
# Partially parsed test_persistent_with_clean_and_instance. Retrieved 27/29 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 24/26 statements.
# Partially parsed test_persistent_with_invariant_error_codes. Retrieved 23/25 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 17/25 statements.


import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = 'MockPMap'
    var_14 = ()
    var_15 = '_buckets'
    var_16 = '_size'
    var_17 = 'mock_buckets'
    var_18 = 'mock_size'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_13, var_14, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1._PRecordEvolver(var_12, var_23)
    var_25 = True
    var_26 = var_24.persistent()
    var_27 = isinstance(var_26, var_12)
    var_28 = bool(var_27)
    assert var_28 is True
    var_29 = var_26._precord_buckets
    assert var_29 == 'mock_buckets'
    var_30 = var_26._precord_size
    assert var_30 == 'mock_size'

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = 'MockPMap'
    var_14 = ()
    var_15 = '_buckets'
    var_16 = '_size'
    var_17 = 'mock_buckets'
    var_18 = 'mock_size'
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_13, var_14, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1._PRecordEvolver(var_12, var_23)
    var_25 = False
    var_26 = ()
    var_27 = {var_15: var_17, var_16: var_18}
    var_28 = [var_13, var_26, var_27]
    var_29 = {}
    var_30 = module_0.type(*var_28, **var_29)
    var_31 = var_30()
    var_32 = var_24.persistent()
    var_33 = bool(var_32 == var_31)
    assert var_33 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = 'field1'
    var_8 = {var_7}
    var_9 = []
    var_10 = {var_2: var_6, var_3: var_8, var_4: var_9, var_5: var_0}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = 'MockPMap'
    var_15 = ()
    var_16 = '_buckets'
    var_17 = '_size'
    var_18 = {}
    var_19 = 0
    var_20 = {var_16: var_18, var_17: var_19}
    var_21 = [var_14, var_15, var_20]
    var_22 = {}
    var_23 = module_0.type(*var_21, **var_22)
    var_24 = var_23()
    var_25 = module_1._PRecordEvolver(var_13, var_24)
    var_26 = True
    var_27 = var_25.persistent()
    var_28 = bool(False)
    assert var_28 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = 'MockPMap'
    var_14 = ()
    var_15 = '_buckets'
    var_16 = '_size'
    var_17 = {}
    var_18 = 0
    var_19 = {var_15: var_17, var_16: var_18}
    var_20 = [var_13, var_14, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = module_1._PRecordEvolver(var_12, var_23)
    var_25 = 'error1'
    var_26 = var_24.persistent()
    var_27 = bool(False)
    assert var_27 is True

import builtins as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = 'MockPMap'
    var_9 = ()
    var_10 = '_buckets'
    var_11 = '_size'
    var_12 = {}
    var_13 = 0
    var_14 = {var_10: var_12, var_11: var_13}
    var_15 = [var_8, var_9, var_14]
    var_16 = {}
    var_17 = module_0.type(*var_15, **var_16)
    var_18 = var_17()
    var_19 = bool(False)
    assert var_19 is True



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_predicate_false.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 12/13 statements.
# Partially parsed test_precord_new_without_special_attributes. Retrieved 5/10 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 5/10 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 7/11 statements.


def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = 'b'
    var_8 = (var_7, var_2)
    var_9 = (var_8,)
    var_10 = [var_6, var_9]
    var_11 = {var_0: var_2, var_1: var_10}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 10
    var_4 = 20

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2
    var_5 = {var_0}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_precord_constructor_with_valid_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 9/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : 1
    var_1 = lambda : 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = 'z'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_precord_fields_initialized.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_missing_fields_added_when_mandatory_fields_not_present. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'TestRecord'
    var_6 = []
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = 'field1'
    var_10 = 'value1'
    var_11 = 'TestRecord.field2'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'test'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 13/15 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 7/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = None
    var_4 = 'a'
    var_5 = 1
    var_6 = (var_4, var_5)
    var_7 = [var_6]
    var_8 = 'b'
    var_9 = (var_8, var_2)
    var_10 = [var_9]
    var_11 = [var_3, var_7, var_3, var_10]
    var_12 = {var_0: var_2, var_1: var_11}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '_factory_fields'
    var_3 = 1
    var_4 = 2
    var_5 = [var_0]
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '_ignore_extra'
    var_3 = 1
    var_4 = 2
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_5: var_6}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = lambda : 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_5: var_6}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value1'
    var_3 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: x.upper()
    var_1 = module_0.field(serializer=var_0)
    var_2 = lambda x: x * 2
    var_3 = module_0.field(serializer=var_2)
    var_4 = 'value1'
    var_5 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x, fmt=None: f'{x}_{fmt}' if fmt else x
    var_1 = module_0.field(serializer=var_0)
    var_2 = lambda x, fmt=None: x if fmt else x.upper()
    var_3 = module_0.field(serializer=var_2)
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'json'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 5/9 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 3/6 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 5/9 statements.
# Partially parsed test_persistent_with_invariant_error. Retrieved 5/9 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = None
    var_4 = False
    var_5 = 'field1'
    var_6 = 'value1'

def test_case_0():
    var_0 = 'value1'
    var_1 = None
    var_2 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = None
    var_4 = False
    var_5 = 'field1'
    var_6 = 'value1'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TestPRecord.mandatory_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = None
    var_4 = False
    var_5 = 'field1'
    var_6 = 'invalid_value'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'INVALID_VALUE'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = None
    var_4 = False
    var_5 = 'field1'
    var_6 = 'value1'
    var_7 = 'field2'
    var_8 = 'value2'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'GLOBAL_INVARIANT_FAILED'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_precord_meta_new_with_no_bases_and_no_fields. Retrieved 6/13 statements.
# Failed to parse test_precord_meta_new_with_fields.
# Failed to parse test_precord_meta_new_with_inherited_fields.
# Partially parsed test_precord_meta_new_with_invariants. Retrieved 5/4 statements.
# Partially parsed test_precord_meta_new_with_inherited_invariants. Retrieved 4/4 statements.


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = set()
    var_4 = '_precord_initial_values'
    var_5 = '__slots__'

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = 'Test OK'
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 2

def test_case_0():
    var_0 = True
    var_1 = 'Test OK'
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 2

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 8/14 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 9/13 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 1
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_1, var_5]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = lambda : var_3
    var_5 = 'c'
    var_6 = 3
    var_7 = 4
    var_8 = {var_1: var_6, var_5: var_7}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_precord_repr_format. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_field_exists. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = []
    var_2 = {}
    var_3 = module_0.PMap(*var_1, **var_2)
    var_4 = 'key'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_missing_fields_are_added_when_mandatory_fields_exist. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'TestClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_precord_mandatory_fields_is_subset_of_precord_fields.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_persistent_global_invariant_failure. Retrieved 9/17 statements.


import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockPRecord'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = module_2._PRecordEvolver(var_12, var_15)
    var_17 = var_16.persistent()
    var_18 = isinstance(var_17, var_12)
    var_19 = bool(var_18)
    assert var_19 is True
    var_20 = bool(var_17 == var_15)
    assert var_20 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockPRecord'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = module_2._PRecordEvolver(var_12, var_15)
    var_17 = 'key'
    var_18 = 'value'
    var_19 = var_16.set(var_17, var_18)
    var_20 = var_16.persistent()
    var_21 = isinstance(var_20, var_12)
    var_22 = bool(var_21)
    assert var_22 is True
    var_23 = bool(var_20 != var_15)
    assert var_23 is True
    var_24 = var_20['key']
    assert var_24 == 'value'

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockPRecord'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = 'mandatory_field'
    var_8 = {var_7}
    var_9 = []
    var_10 = {var_2: var_6, var_3: var_8, var_4: var_9, var_5: var_0}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = []
    var_15 = {}
    var_16 = module_1.PMap(*var_14, **var_15)
    var_17 = module_2._PRecordEvolver(var_13, var_16)
    var_18 = var_17.persistent()
    var_19 = bool(False)
    assert var_19 is True
    var_20 = 'MockPRecord.mandatory_field'

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockPRecord'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = lambda x: failing_invariant(x)
    var_9 = [var_8]
    var_10 = {var_2: var_6, var_3: var_7, var_4: var_9, var_5: var_0}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = []
    var_15 = {}
    var_16 = module_1.PMap(*var_14, **var_15)
    var_17 = module_2._PRecordEvolver(var_13, var_16)
    var_18 = 'key'
    var_19 = 'value'
    var_20 = var_17.set(var_18, var_19)
    var_21 = var_17.persistent()
    var_22 = bool(False)
    assert var_22 is True
    var_23 = 'INVARIANT_FAILED'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockPRecord'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {}
    var_10 = module_0.PMap(*var_8, **var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'GLOBAL_INVARIANT_FAILED'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_persistent_raises_when_invariant_error_codes_or_missing_fields. Retrieved 6/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'error1'
    var_7 = 'field1'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 5/11 statements.
# Partially parsed test_set_with_valid_field_and_factory_ignore_extra. Retrieved 6/12 statements.
# Partially parsed test_set_with_invalid_field_type. Retrieved 5/12 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 5/11 statements.
# Partially parsed test_set_with_missing_field. Retrieved 4/8 statements.
# Partially parsed test_set_with_factory_field_not_in_factory_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value: value * 2
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'field1'
    var_7 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value, ignore_extra=False: value * 2 if not ignore_extra else value * 3
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = True
    var_7 = 'field1'
    var_8 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value: value
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'field1'
    var_7 = 'not_an_int'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (False, 'INVALID') if value < 0 else (True, None)
    var_1 = lambda value: value
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'field1'
    var_7 = -5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = {}
    var_3 = module_0.PMap(*var_1, **var_2)
    var_4 = 'nonexistent_field'
    var_5 = 5
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = lambda value: value * 2
    var_2 = 'field1'
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = []
    var_7 = 'field1'
    var_8 = 5



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_repr_format. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = 'TestRecord('
    var_3 = ')'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_missing_mandatory_fields_are_added_to_missing_fields. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'TestClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_persistent_predicate_false. Retrieved 6/17 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = {}
    var_3 = set()
    var_4 = []
    var_5 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 9/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 4/5 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 1
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = '_precord_size'
    var_10 = '_precord_buckets'
    var_11 = {var_9: var_2, var_10: var_6}
    var_12 = module_0.PRecord(*var_8, **var_11)
    var_13 = var_12['field1']
    assert var_13 == 'value1'

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'field1'
    var_7 = '_factory_fields'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.PRecord(*var_5, **var_8)
    var_10 = var_9['field1']
    assert var_10 == 'value1'

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = 'field1'
    var_9 = 'extra_field'
    var_10 = '_ignore_extra'
    var_11 = {var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.PRecord(*var_7, **var_11)
    var_13 = var_12['field1']
    assert var_13 == 'value1'
    var_14 = 'extra_field'
    var_15 = bool('extra_field' not in var_12)
    assert var_15 is True

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'field1'
    var_5 = {var_4: var_1}
    var_6 = module_0.PRecord(*var_3, **var_5)
    var_7 = var_6['field1']
    assert var_7 == 'value1'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 9/10 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = (var_2, var_5)
    var_7 = None
    var_8 = [var_6, var_7, var_7, var_7]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'test_key'
    var_7 = 'test_value'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 9/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/6 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 10
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = []
    var_9 = '_precord_size'
    var_10 = '_precord_buckets'
    var_11 = {var_9: var_2, var_10: var_6}
    var_12 = module_0.PRecord(*var_8, **var_11)
    var_13 = var_12['key']
    assert var_13 == 'value'

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'field1'
    var_7 = '_factory_fields'
    var_8 = {var_6: var_2, var_7: var_3}
    var_9 = module_0.PRecord(*var_5, **var_8)
    var_10 = var_9['field1']
    assert var_10 == 'value1'

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = []
    var_8 = 'field1'
    var_9 = 'extra_field'
    var_10 = '_ignore_extra'
    var_11 = {var_8: var_3, var_9: var_4, var_10: var_5}
    var_12 = module_0.PRecord(*var_7, **var_11)
    var_13 = var_12['field1']
    assert var_13 == 'value1'
    var_14 = 'extra_field'
    var_15 = bool('extra_field' not in var_12)
    assert var_15 is True

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'initial_value'
    var_2 = 'updated_value'
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'field1'
    var_6 = {var_5: var_2}
    var_7 = module_0.PRecord(*var_4, **var_6)
    var_8 = var_7['field1']
    assert var_8 == 'updated_value'

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = []
    var_4 = {}
    var_5 = module_0.PRecord(*var_3, **var_4)
    var_6 = var_5['field1']
    assert var_6 == 'computed_value'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_instance. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_clean_and_instance. Retrieved 1/5 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 1/4 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field1'
    var_4 = 'value1'

def test_case_0():
    var_0 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'TestPRecord.mandatory_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field_with_invariant'
    var_4 = 'invalid_value'
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_persistent_with_invariant_error_codes. Retrieved 8/16 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 8/16 statements.


import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = []
    var_13 = {}
    var_14 = module_1.PMap(*var_12, **var_13)
    var_15 = module_2._PRecordEvolver(var_11, var_14)
    var_16 = 'key'
    var_17 = 'value'
    var_18 = var_15.set(var_16, var_17)
    var_19 = var_15.persistent()
    var_20 = isinstance(var_19, var_11)
    var_21 = bool(var_20)
    assert var_21 is True
    var_22 = var_19._precord_buckets
    var_23 = bool(var_19._precord_buckets == var_15._buckets)
    assert var_23 is True
    var_24 = var_19._precord_size
    var_25 = bool(var_19._precord_size == var_15._size)
    assert var_25 is True

import builtins as module_0
import pyrsistent._precord as module_1

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = {}
    var_13 = 0
    var_14 = var_11(_precord_buckets=var_12, _precord_size=var_13)
    var_15 = module_1._PRecordEvolver(var_11, var_14)
    var_16 = var_15.persistent()
    var_17 = bool(var_16 is var_14)
    assert var_17 is True

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = [var_0, var_1, var_9]
    var_11 = {}
    var_12 = module_0.type(*var_10, **var_11)
    var_13 = []
    var_14 = {}
    var_15 = module_1.PMap(*var_13, **var_14)
    var_16 = module_2._PRecordEvolver(var_12, var_15)
    var_17 = var_16.persistent()
    var_18 = bool(False)
    assert var_18 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 4/5 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 5/9 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 6/10 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = [var_1]
    var_3 = var_2 * var_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'b'
    var_6 = 3
    var_7 = {var_5: var_6}



# Parsed testcases at query #4
#--------------------------




def test_case_0():
    var_0 = bool(not ('_precord_size' in {'a': 1} and '_precord_buckets' in {'a': 1}))
    assert var_0 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_new_with_no_bases_and_empty_dct. Retrieved 4/5 statements.
# Partially parsed test_new_with_bases_and_fields. Retrieved 2/6 statements.
# Partially parsed test_new_with_invariant. Retrieved 5/2 statements.
# Partially parsed test_new_with_inherited_invariant. Retrieved 4/11 statements.
# Partially parsed test_new_with_non_callable_invariant_raises. Retrieved 5/7 statements.
# Partially parsed test_new_with_multiple_invariants. Retrieved 2/17 statements.
# Partially parsed test_new_with_field_in_dct. Retrieved 4/7 statements.
# Partially parsed test_new_with_initial_value_none. Retrieved 4/7 statements.
# Partially parsed test_new_with_initial_value_no_initial. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = set()

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 'TestClass'
    var_3 = {}
    var_4 = 'x'

def test_case_0():
    var_0 = True
    var_1 = 'TestClass'
    var_2 = ()
    var_3 = '__invariant__'
    var_4 = 0

def test_case_0():
    var_0 = True
    var_1 = 'TestClass'
    var_2 = ()
    var_3 = '__invariant__'
    var_4 = 0

def test_case_0():
    var_0 = lambda : True
    var_1 = 'TestClass'
    var_2 = {}
    var_3 = 0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '__invariant__'
    var_3 = 'not callable'
    var_4 = {var_2: var_3}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'x'
    var_3 = True
    var_4 = 'x'
    var_5 = 'x'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'x'
    var_3 = None

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'x'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_set_existing_field_with_valid_value. Retrieved 17/25 statements.
# Partially parsed test_set_existing_field_with_invalid_type. Retrieved 18/27 statements.
# Partially parsed test_set_existing_field_with_invariant_failure. Retrieved 17/25 statements.
# Partially parsed test_set_existing_field_with_factory_and_ignore_extra. Retrieved 17/26 statements.
# Partially parsed test_set_existing_field_with_factory_and_invariant_exception. Retrieved 20/29 statements.
# Partially parsed test_set_existing_field_not_in_factory_fields. Retrieved 18/26 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = '__name__'
    var_14 = 'field1'
    var_15 = []
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = '__name__'
    var_14 = 'field1'
    var_15 = []
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = 'field1'
    var_19 = 'not_an_int'
    var_20 = bool(False)
    assert var_20 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = False
    var_7 = 'INVALID'
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = '__name__'
    var_14 = 'field1'
    var_15 = []
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x, ignore_extra=False: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = '__name__'
    var_14 = 'field1'
    var_15 = []
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = 1
    var_6 = 0
    var_7 = var_5 / var_6
    var_8 = lambda x: var_7
    var_9 = True
    var_10 = None
    var_11 = (var_9, var_10)
    var_12 = lambda x: var_11
    var_13 = 'TestClass'
    var_14 = ()
    var_15 = '_precord_fields'
    var_16 = '__name__'
    var_17 = 'field1'
    var_18 = []
    var_19 = {}
    var_20 = module_0.PMap(*var_18, **var_19)
    var_21 = 10

import builtins as module_0
import pyrsistent._pmap as module_1
import pyrsistent._precord as module_2

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '__name__'
    var_4 = {}
    var_5 = {var_2: var_4, var_3: var_0}
    var_6 = [var_0, var_1, var_5]
    var_7 = {}
    var_8 = module_0.type(*var_6, **var_7)
    var_9 = []
    var_10 = {}
    var_11 = module_1.PMap(*var_9, **var_10)
    var_12 = module_2._PRecordEvolver(var_8, var_11)
    var_13 = 'nonexistent'
    var_14 = 10
    var_15 = var_12.set(var_13, var_14)
    var_16 = bool(False)
    assert var_16 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = '__name__'
    var_14 = 'field1'
    var_15 = []
    var_16 = {}
    var_17 = module_0.PMap(*var_15, **var_16)
    var_18 = set()
    var_19 = 10



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_missing_fields_added_when_mandatory_fields_not_present. Retrieved 5/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'MockPRecord'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = 'MockPRecord.field2'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'error1'
    var_7 = 'missing1'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'test_key'
    var_7 = 'test_value'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_missing_fields_added_when_mandatory_fields_exist. Retrieved 7/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'TestClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = [var_9]
    var_11 = {}
    var_12 = module_0.PMap(*var_10, **var_11)



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_predicate_line_1.




# Parsed testcases at query #12
#--------------------------

# Partially parsed test_precord_initial_values_used. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = lambda : 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_repr_empty_record.
# Partially parsed test_repr_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_repr_with_complex_values. Retrieved 8/11 statements.
# Partially parsed test_repr_with_escaped_strings. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'value1'
    var_4 = 42
    var_5 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Hello\nWorld'



