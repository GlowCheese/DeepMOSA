####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_same_object_when_not_dirty_and_is_instance. Retrieved 6/19 statements.
# Partially parsed test_persistent_returns_new_object_when_dirty. Retrieved 7/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_failure. Retrieved 8/24 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 6/22 statements.
# Partially parsed test_persistent_raises_global_invariant_exception. Retrieved 6/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = [var_5]
    var_7 = {}
    var_8 = module_0.PMap(*var_6, **var_7)
    var_9 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = True
    var_2 = set()
    var_3 = []
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = 2

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.PMap(*var_5, **var_6)
    var_8 = 'a'
    var_9 = 1
    var_10 = 'MockPRecord.a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = lambda x: (False, 'GLOBAL_ERR')
    var_3 = [var_2]
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = [var_6]
    var_8 = {}
    var_9 = module_0.PMap(*var_7, **var_8)
    var_10 = 'Global invariant failed'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_PRecordMeta_new.




# Parsed testcases at query #3
#--------------------------

# Partially parsed test_persistent_is_dirty_evaluates_to_true. Retrieved 4/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_persistent_not_dirty_and_is_instance_of_cls. Retrieved 6/20 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_repr_basic_fields. Retrieved 2/7 statements.
# Partially parsed test_repr_with_extra_fields_ignored. Retrieved 6/10 statements.
# Failed to parse test_repr_empty_record.
# Partially parsed test_repr_complex_types. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'name'
    var_1 = 'extra'
    var_2 = 'Bob'
    var_3 = 'data'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'id'
    var_4 = {var_3: var_0}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_set_valid_field_updates_value. Retrieved 3/16 statements.
# Partially parsed test_set_invalid_field_raises_attribute_error. Retrieved 5/22 statements.
# Partially parsed test_set_type_mismatch_raises_p_type_error. Retrieved 5/18 statements.
# Partially parsed test_set_invariant_failure_records_error. Retrieved 3/22 statements.
# Partially parsed test_set_with_factory_fields_filtering. Retrieved 5/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field_a'
    var_4 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'non_existent_field'
    var_4 = 10
    var_5 = "'non_existent_field' is not among the specified fields for MockRecord"
    var_6 = 'AttributeError not raised'
    var_7 = AssertionError(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field_a'
    var_4 = 'not an int'
    var_5 = 'Invalid type for field MockRecord.field_a'
    var_6 = 'PTypeError not raised for type mismatch'
    var_7 = AssertionError(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field_a'
    var_4 = -1
    var_5 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.PMap(*var_0, **var_1)
    var_3 = 'field_a'
    var_4 = {var_3}
    var_5 = 10
    var_6 = 'field_b'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_repr_format. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {}
    var_6 = 1
    var_7 = 'test'



# Parsed testcases at query #8
#--------------------------




import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = None
    var_4 = '_precord_fields'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = '__slots__'
    var_8 = 'f1'
    var_9 = 'f1'
    var_10 = 'f1'
    var_11 = 'f2'
    var_12 = 'f2'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_basic_functionality. Retrieved 4/13 statements.
# Partially parsed test_serialize_with_different_format. Retrieved 3/13 statements.
# Partially parsed test_serialize_empty_record. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = 1
    var_4 = 2
    var_5 = 'json'

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {}
    var_3 = 'data'
    var_4 = 'xml'

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'any'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_exist. Retrieved 3/29 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_exist. Retrieved 4/30 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = 'a'
    var_4 = {var_3}
    var_5 = []
    var_6 = {}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_prerecord_new_with_internal_args. Retrieved 8/22 statements.
# Partially parsed test_prerecord_new_with_factory_fields_and_ignore_extra. Retrieved 7/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = 10
    var_5 = None
    var_6 = [var_5]
    var_7 = var_6 * var_4

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = 1
    var_5 = 'a'
    var_6 = [var_5]
    var_7 = True



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_same_object_if_not_dirty_and_is_correct_type. Retrieved 2/31 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_error_codes. Retrieved 5/34 statements.
# Partially parsed test_persistent_raises_missing_fields_error. Retrieved 4/29 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = {}
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}
    var_5 = 'MockPRecord.a'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_new_with_kwargs. Retrieved 1/2 statements.
# Partially parsed test_precord_new_overrides_initial_values. Retrieved 2/3 statements.
# Partially parsed test_precord_new_internal_bypass. Retrieved 8/9 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/12 statements.
# Partially parsed test_precord_new_ignore_extra_flag. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 2
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = [var_6]

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = 5
    var_5 = 'a'
    var_6 = {var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = 1
    var_5 = 'not_here'
    var_6 = True
    var_7 = 'a'
    var_8 = 'extra'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_PRecordMeta__new_basic_inheritance. Retrieved 4/12 statements.
# Failed to parse test_PRecordMeta__new_with_fields_and_invariants.


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_PRecordMeta__new__. Retrieved 1/6 statements.


def test_case_0():
    var_0 = lambda x: (False, ('error',))
    var_1 = 'f1'
    var_2 = 'f2'
    var_3 = 'f3'
    var_4 = 'f1'
    var_5 = 'f3'
    var_6 = 'f2'
    var_7 = 'f2'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_set_updates_value_and_returns_self. Retrieved 5/24 statements.
# Partially parsed test_set_raises_attribute_error_for_missing_field. Retrieved 5/19 statements.
# Partially parsed test_set_handles_invariant_exception_and_stores_errors. Retrieved 5/24 statements.
# Partially parsed test_set_applies_invariant_check_and_records_error_code. Retrieved 5/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'b'
    var_7 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 10
    var_8 = 'ERR_01'
    var_9 = 'field.a'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 10
    var_8 = 'INVALID_VALUE'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_repr_basic_functionality. Retrieved 2/7 statements.
# Failed to parse test_repr_empty_record.
# Partially parsed test_repr_with_different_types. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_constructor_with_values. Retrieved 2/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_with_extra_fields_kept_by_default. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Unknown'
    var_3 = 0
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 25

def test_case_0():
    var_0 = 'val'
    var_1 = lambda : 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'name'
    var_1 = 'extra'
    var_2 = 'Alice'
    var_3 = 'data'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = 'name'
    var_7 = 'extra'

def test_case_0():
    var_0 = 'Alice'
    var_1 = 'data'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_set_updates_value_and_returns_self_on_valid_field. Retrieved 5/20 statements.
# Partially parsed test_set_raises_attribute_error_for_non_existent_field. Retrieved 5/17 statements.
# Partially parsed test_set_handles_invariant_exception_during_factory. Retrieved 4/19 statements.
# Partially parsed test_set_applies_invariant_check_and_records_error. Retrieved 5/15 statements.
# Partially parsed test_set_skips_factory_if_field_not_in_factory_fields. Retrieved 8/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'b'
    var_7 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.PMap(*var_2, **var_3)
    var_5 = 'a'
    var_6 = 10
    var_7 = 'err'
    var_8 = 'field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'a'
    var_7 = 10
    var_8 = 'error_code'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = set()
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'a'
    var_8 = None
    var_9 = {var_7: var_8}
    var_10 = 10
    var_11 = 'b'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_p_record_new_does_not_trigger_hack_total_path. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_PRecordMeta_new_basic_functionality. Retrieved 1/13 statements.
# Failed to parse test_PRecordMeta_new_with_invariants_inheritance.


def test_case_0():
    var_0 = lambda x: True
    var_1 = True
    var_2 = 10
    var_3 = False
    var_4 = 20
    var_5 = True
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = 'field3'
    var_9 = 'field1'
    var_10 = 'field3'
    var_11 = 'field2'
    var_12 = 'field3'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_set_field_exists_evaluates_to_true. Retrieved 4/24 statements.


def test_case_0():
    var_0 = 'test_key'
    var_1 = set()
    var_2 = []
    var_3 = 'test_key'
    var_4 = 'some_value'



# Parsed testcases at query #12
#--------------------------




def test_case_0():
    var_0 = bool(True)
    assert var_0 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_precord_new_with_initial_values. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 20



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_new_with_internal_args_bypass_logic. Retrieved 8/17 statements.
# Partially parsed test_precord_new_standard_construction_flow. Retrieved 6/11 statements.
# Partially parsed test_precord_new_with_initial_values_and_factory. Retrieved 4/8 statements.
# Partially parsed test_precord_new_with_factory_fields_filtering. Retrieved 9/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = 10
    var_5 = None
    var_6 = [var_5]
    var_7 = var_6 * var_4

import builtins as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Field'
    var_3 = ()
    var_4 = 'factory'
    var_5 = 'invariant'
    var_6 = lambda x: x
    var_7 = lambda x: (True, None)
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = [var_2, var_3, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = 'Field'
    var_13 = ()
    var_14 = 'factory'
    var_15 = 'invariant'
    var_16 = lambda x: x
    var_17 = lambda x: (True, None)
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = [var_12, var_13, var_18]
    var_20 = {}
    var_21 = module_0.type(*var_19, **var_20)
    var_22 = {var_0: var_11, var_1: var_21}
    var_23 = {}
    var_24 = set()
    var_25 = []
    var_26 = 'Alice'
    var_27 = 30

import builtins as module_0

def test_case_0():
    var_0 = 'val'
    var_1 = 'Field'
    var_2 = ()
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = lambda x: (True, None)
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = [var_1, var_2, var_7]
    var_9 = {}
    var_10 = module_0.type(*var_8, **var_9)
    var_11 = {var_0: var_10}
    var_12 = 'val'
    var_13 = lambda : 10
    var_14 = {var_12: var_13}
    var_15 = set()
    var_16 = []

import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'factory'
    var_3 = 'invariant'
    var_4 = lambda x: x + 1
    var_5 = lambda x: (True, None)
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = {}
    var_9 = module_0.type(*var_7, **var_8)
    var_10 = 'Field'
    var_11 = ()
    var_12 = 'factory'
    var_13 = 'invariant'
    var_14 = lambda x: x + 1
    var_15 = lambda x: (True, None)
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = [var_10, var_11, var_16]
    var_18 = {}
    var_19 = module_0.type(*var_17, **var_18)
    var_20 = 'a'
    var_21 = 'b'
    var_22 = 'Field'
    var_23 = ()
    var_24 = 'factory'
    var_25 = 'invariant'
    var_26 = lambda x: x + 1
    var_27 = lambda x: (True, None)
    var_28 = {var_24: var_26, var_25: var_27}
    var_29 = [var_22, var_23, var_28]
    var_30 = {}
    var_31 = module_0.type(*var_29, **var_30)
    var_32 = 'Field'
    var_33 = ()
    var_34 = 'factory'
    var_35 = 'invariant'
    var_36 = lambda x: x + 1
    var_37 = lambda x: (True, None)
    var_38 = {var_34: var_36, var_35: var_37}
    var_39 = [var_32, var_33, var_38]
    var_40 = {}
    var_41 = module_0.type(*var_39, **var_40)
    var_42 = {var_20: var_31, var_21: var_41}
    var_43 = {}
    var_44 = set()
    var_45 = []
    var_46 = 'a'
    var_47 = True
    var_48 = {var_46: var_47}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_basic_values. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/13 statements.
# Partially parsed test_serialize_returns_dict. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'alice'
    var_1 = 'upper'
    var_2 = 'lower'

def test_case_0():
    var_0 = 10
    var_1 = 'val'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_basic_functionality. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_argument. Retrieved 4/16 statements.
# Partially parsed test_serialize_ignores_unmapped_fields_in_output. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 'test'
    var_3 = 'upper'
    var_4 = 'none'

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = 1
    var_3 = 99
    var_4 = 'a'
    var_5 = 'extra'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_exist. Retrieved 6/24 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_exist. Retrieved 5/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = {}
    var_5 = module_0.PMap(*var_3, **var_4)
    var_6 = 'error_code_1'
    var_7 = 'InvariantException was not raised'
    var_8 = AssertionError(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = []
    var_4 = []
    var_5 = {}
    var_6 = module_0.PMap(*var_4, **var_5)
    var_7 = 'InvariantException was not raised'
    var_8 = AssertionError(var_7)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_precord_constructor_with_values. Retrieved 2/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_overriding_initial_values. Retrieved 3/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/3 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_with_extra_fields_kept. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 3
    var_1 = 10

def test_case_0():
    var_0 = 3
    var_1 = 10
    var_2 = 5

def test_case_0():
    var_0 = lambda : 0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = 'a'
    var_7 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = False



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_exist. Retrieved 3/27 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_exist. Retrieved 4/28 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = set()
    var_2 = []
    var_3 = None

def test_case_0():
    var_0 = 'a'
    var_1 = 'a'
    var_2 = {var_1}
    var_3 = 'a'
    var_4 = {var_3}
    var_5 = []
    var_6 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_precord_constructor_basic_initialization. Retrieved 2/7 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 1/8 statements.
# Partially parsed test_precord_constructor_ignores_extra_fields_when_flag_set. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_preserves_extra_fields_by_default. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'Alice'
    var_1 = 30

def test_case_0():
    var_0 = 'retries'
    var_1 = 'timeout'
    var_2 = 3
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True
    var_6 = 'a'
    var_7 = 'b'

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_PRecordMeta__new_sets_fields_and_invariants. Retrieved 2/12 statements.


def test_case_0():
    var_0 = lambda x: (True, 'data')
    var_1 = lambda x: (True, 'new_data')
    var_2 = '_precord_fields'
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = '_precord_initial_values'
    var_7 = '__slots__'



