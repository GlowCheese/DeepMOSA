####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_same_instance_if_not_dirty. Retrieved 5/16 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_failure. Retrieved 5/21 statements.
# Partially parsed test_persistent_detects_missing_mandatory_fields. Retrieved 3/13 statements.
# Partially parsed test_persistent_triggers_global_invariants. Retrieved 6/28 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = bool(True)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = set()
    var_3 = lambda x: (False, 'GLOBAL_ERR')
    var_4 = [var_3]
    var_5 = 'pyrsistent._field_common'
    var_6 = []
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = 'a'
    var_10 = 1
    var_11 = bool(True)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_test___new__. Retrieved 2/3 statements.
# Partially parsed test_test___new___with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_test___new___with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_test___new___internal_reconstruction. Retrieved 9/10 statements.
# Partially parsed test_test___new___ignore_extra_logic. Retrieved 3/4 statements.


def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 100
    var_3 = 200
    var_4 = 'a'
    var_5 = [var_4]

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_6]
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_PRecordMeta__new__. Retrieved 15/29 statements.


def test_case_0():
    var_0 = 'TestRecord'
    var_1 = True
    var_2 = 'default_value'
    var_3 = '_precord_fields'
    var_4 = '__invariant__'
    var_5 = 'a'
    var_6 = 'b'
    var_7 = False
    var_8 = 'PFIELD_NO_INITIAL'
    var_9 = True
    var_10 = ()
    var_11 = (var_9, var_10)
    var_12 = '_precord_fields'
    var_13 = '_precord_invariants'
    var_14 = '__invariant__'
    var_15 = '_precord_mandatory_fields'
    var_16 = '_precord_initial_values'
    var_17 = 'a'
    var_18 = 'b'
    var_19 = 'b'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_persistent_success. Retrieved 5/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_error. Retrieved 7/22 statements.
# Partially parsed test_persistent_raises_missing_fields. Retrieved 6/15 statements.
# Partially parsed test_persistent_raises_global_invariant_exception. Retrieved 4/27 statements.
# Partially parsed test_persistent_attribute_error_on_set. Retrieved 6/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'ERR_A'
    var_9 = 'Expected InvariantException'
    var_10 = AssertionError(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'MockPRecord.a'
    var_8 = 'Expected InvariantException for missing field'
    var_9 = AssertionError(var_8)

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = lambda x: (False, 'GLOBAL_ERR')
    var_3 = [var_2]
    var_4 = 'GLOBAL_ERR'
    var_5 = 'Expected global invariant failure'
    var_6 = AssertionError(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'unknown'
    var_7 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_precord_repr_basic. Retrieved 2/7 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_with_complex_types. Retrieved 5/12 statements.
# Partially parsed test_precord_repr_order_consistency. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'id'
    var_4 = {var_3: var_0}

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_persistent_success. Retrieved 5/20 statements.
# Partially parsed test_persistent_raises_missing_fields. Retrieved 3/19 statements.
# Partially parsed test_persistent_raises_invariant_error. Retrieved 5/22 statements.
# Partially parsed test_persistent_raises_global_invariant. Retrieved 5/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'MockPRecord.a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 1
    var_8 = 'ERR_01'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = lambda x: (False, 'GLOBAL_ERR')
    var_3 = [var_2]
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = 'GLOBAL_ERR'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_persistent_returns_new_instance_when_dirty. Retrieved 3/36 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_persistent_skips_mandatory_fields_check_when_none. Retrieved 5/55 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = False
    var_4 = 'test_field'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_present. Retrieved 4/18 statements.
# Partially parsed test_persistent_raises_exception_on_invariant_error. Retrieved 4/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'ERR_001'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'ERROR_CODE'



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_same_instance_if_not_dirty. Retrieved 2/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_error. Retrieved 2/14 statements.
# Partially parsed test_persistent_detects_missing_mandatory_fields. Retrieved 1/15 statements.
# Partially parsed test_persistent_calls_global_invariants. Retrieved 7/28 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = []
    var_2 = {}
    var_3 = module_0.PMap(*var_1, **var_2)

def test_case_0():
    var_0 = 'a'

def test_case_0():
    var_0 = 'a'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'a'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_new_with_valid_kwargs. Retrieved 2/3 statements.
# Partially parsed test_precord_new_with_missing_mandatory_fields_raises_error. Retrieved 1/2 statements.
# Partially parsed test_precord_new_with_internal_bypass_logic. Retrieved 7/8 statements.
# Partially parsed test_precord_new_with_initial_values_logic. Retrieved 2/4 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 4/5 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/5 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'Alice'

def test_case_0():
    var_0 = None
    var_1 = 'name'
    var_2 = 'Bob'
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 2

def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'age'
    var_1 = [var_0]
    var_2 = 'Alice'
    var_3 = '30'

def test_case_0():
    var_0 = True
    var_1 = 'Alice'
    var_2 = 30
    var_3 = 'data'
    var_4 = 'name'
    var_5 = 'unknown_field'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_on_field_error. Retrieved 4/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'err_1'
    var_7 = 'err_1'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_repr_basic. Retrieved 2/7 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_complex_types. Retrieved 4/10 statements.
# Partially parsed test_precord_repr_order_consistency. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_precord_new_with_valid_kwargs. Retrieved 2/3 statements.
# Partially parsed test_precord_new_with_default_values. Retrieved 1/2 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/5 statements.
# Partially parsed test_precord_new_internal_reconstruction. Retrieved 7/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2}
    var_4 = 10
    var_5 = 20

def test_case_0():
    var_0 = True
    var_1 = 10
    var_2 = 20
    var_3 = 'not_a_field'
    var_4 = 'extra'

def test_case_0():
    var_0 = None
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = [var_0, var_4]
    var_6 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_persistent_skips_mandatory_fields_check_when_no_mandatory_fields_exist. Retrieved 3/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_new_basic_initialization. Retrieved 2/3 statements.
# Partially parsed test_precord_new_with_defaults_via_initial_values. Retrieved 2/3 statements.
# Partially parsed test_precord_new_internal_reconstruction. Retrieved 8/9 statements.
# Partially parsed test_precord_new_factory_fields_filtering. Retrieved 4/8 statements.
# Partially parsed test_precord_new_ignore_extra_logic. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'Bob'
    var_1 = 25

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Charlie'
    var_3 = 40
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = len(var_5)
    var_7 = var_5._buckets

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'val_a'
    var_3 = 'val_b'

def test_case_0():
    var_0 = True
    var_1 = 10
    var_2 = 'not_allowed'
    var_3 = 'a'
    var_4 = 'extra'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_persistent_returns_pm_when_not_dirty_and_is_instance. Retrieved 4/33 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'Evolver'
    var_4 = 1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_set_valid_field_updates_value. Retrieved 5/18 statements.
# Partially parsed test_set_invalid_field_type_raises_error. Retrieved 5/18 statements.
# Partially parsed test_set_non_existent_field_raises_attribute_error. Retrieved 6/13 statements.
# Partially parsed test_set_invariant_failure_records_error. Retrieved 5/17 statements.
# Partially parsed test_set_with_factory_fields_filtering. Retrieved 8/21 statements.
# Partially parsed test_set_with_ignore_extra_param. Retrieved 9/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'age'
    var_7 = 25

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'age'
    var_7 = 'not_an_int'
    var_8 = 'Invalid type'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'unknown'
    var_7 = 123
    var_8 = "'unknown' is not among the specified fields"

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'age'
    var_7 = 10
    var_8 = 'ERR_001'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'other'
    var_7 = None
    var_8 = {var_6: var_7}
    var_9 = 'age'
    var_10 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = True
    var_7 = 'age'
    var_8 = 10
    var_9 = evolver_true.persistent()[var_7]
    assert var_9 == 11
    var_10 = False
    var_11 = evolver_false.persistent()[var_7]
    assert var_11 == 10



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_success. Retrieved 6/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_error. Retrieved 8/21 statements.
# Partially parsed test_persistent_raises_missing_fields_error. Retrieved 6/20 statements.
# Partially parsed test_persistent_raises_global_invariant_exception. Retrieved 8/21 statements.
# Partially parsed test_persistent_returns_same_object_if_not_dirty. Retrieved 4/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'a'
    var_8 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'a'
    var_8 = 1
    var_9 = 'ERR_A'
    var_10 = 'InvariantException not raised'
    var_11 = AssertionError(var_10)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockClass'
    var_5 = []
    var_6 = {}
    var_7 = module_0.PMap(*var_5, **var_6)
    var_8 = 'MockClass.a'
    var_9 = 'InvariantException not raised for missing field'
    var_10 = AssertionError(var_9)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = lambda x: (False, 'GLOBAL_ERR')
    var_3 = [var_2]
    var_4 = 'MockClass'
    var_5 = []
    var_6 = {}
    var_7 = module_0.PMap(*var_5, **var_6)
    var_8 = 'a'
    var_9 = 1
    var_10 = 'GLOBAL_ERR'
    var_11 = 'InvariantException not raised for global invariant'
    var_12 = AssertionError(var_11)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_persistent_returns_same_instance_if_not_dirty. Retrieved 5/28 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_error. Retrieved 7/56 statements.
# Partially parsed test_persistent_detects_missing_mandatory_fields. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 'a'
    var_5 = set()
    var_6 = []

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = set()
    var_5 = []
    var_6 = 'a'
    var_7 = set()
    var_8 = []
    var_9 = None
    var_10 = 'ERR_01'

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_persistent_skips_mandatory_field_check_when_no_mandatory_fields_exist. Retrieved 6/55 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_exist. Retrieved 4/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'ERR_001'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_precord_repr_format. Retrieved 3/12 statements.
# Partially parsed test_precord_repr_logic_direct. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30
    var_2 = "User(name='Alice', age=30)"

def test_case_0():
    var_0 = 1
    var_1 = 'test'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_basic_functionality. Retrieved 3/13 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/13 statements.
# Partially parsed test_serialize_with_format_argument. Retrieved 3/15 statements.
# Partially parsed test_serialize_multiple_fields. Retrieved 6/17 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = lambda f, fmt, v: v
    var_2 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = lambda f, fmt, v: str(v)
    var_2 = 100

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'test'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = lambda f, fmt, v: v
    var_3 = 2
    var_4 = lambda f, fmt, v: v * var_3
    var_5 = 1



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serialize_basic_functionality. Retrieved 3/12 statements.
# Partially parsed test_serialize_with_different_format. Retrieved 4/16 statements.
# Partially parsed test_serialize_multiple_fields. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 'test'
    var_3 = 'raw'

def test_case_0():
    var_0 = 'data'
    var_1 = {}
    var_2 = 'value'
    var_3 = 'json'
    var_4 = 'text'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = 1
    var_4 = 'hello'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_field_exists. Retrieved 6/39 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = set()
    var_2 = []
    var_3 = 'pyrsistent'
    var_4 = {}
    var_5 = 'test_key'
    var_6 = 123



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 5/24 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = 'serialize'
    var_3 = None
    var_4 = 123
    var_5 = 'json'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_PRecordMeta_new_logic.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_returns_new_instance_when_dirty. Retrieved 10/58 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = set()
    var_5 = []
    var_6 = 'a'
    var_7 = set()
    var_8 = []
    var_9 = {}
    var_10 = set()
    var_11 = []



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_prerecord_new_skip_hack_branch. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = bool(not ('_precord_size' in {'a': 10} and '_precord_buckets' in {'a': 10}))
    assert var_1 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_set_field_exists_evaluates_to_true. Retrieved 5/25 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test_key'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'test_key'
    var_7 = 'some_value'



