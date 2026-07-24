####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_persistent_returns_instance_of_destination_class.
# Failed to parse test_persistent_raises_invariant_exception_on_missing_mandatory_fields.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_failure. Retrieved 2/9 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 2/11 statements.
# Failed to parse test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 2/8 statements.
# Partially parsed test_persistent_includes_all_set_values. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'field'
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'field'
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'field'
    var_1 = 'new'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = 'b'
    var_3 = 'test'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 7/19 statements.
# Partially parsed test___new___inherits_fields_from_bases. Retrieved 6/23 statements.
# Partially parsed test___new___stores_invariants. Retrieved 5/20 statements.
# Partially parsed test___new___wraps_invariants. Retrieved 5/11 statements.
# Partially parsed test___new___sets_mandatory_fields. Retrieved 6/22 statements.
# Partially parsed test___new___sets_initial_values. Retrieved 7/23 statements.
# Partially parsed test___new___sets_slots. Retrieved 2/3 statements.
# Partially parsed test___new___full_metaclass_creation. Retrieved 1/21 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = False
    var_3 = 10
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = ()
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 'base_field'
    var_3 = 'new_field'
    var_4 = False
    var_5 = 5
    var_6 = '_precord_fields'
    var_7 = 'new_field'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = '_precord_invariants'
    var_2 = 0
    var_3 = None
    var_4 = 1

def test_case_0():
    var_0 = '__invariant__'
    var_1 = ()
    var_2 = '_precord_invariants'
    var_3 = 0
    var_4 = None

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = '_precord_invariants'
    var_5 = '__invariant__'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = False
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = ()
    var_6 = '_precord_fields'

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = False
    var_3 = 10
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = ()
    var_7 = '_precord_fields'

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = var_0['__slots__']
    var_3 = bool(var_0['__slots__'] == ())
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = True
    var_3 = False
    var_4 = 20
    var_5 = '_precord_fields'
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = 'base_field'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_repr_with_empty_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_special_characters_in_field_name. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Bob'
    var_3 = 30
    var_4 = "TestRecord(name='Bob', age=30)"

def test_case_0():
    var_0 = {}
    var_1 = 'TestRecord()'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = "TestRecord(data={'key': 'value'})"

def test_case_0():
    var_0 = 'field_name'
    var_1 = 'test'
    var_2 = "TestRecord(field_name='test')"



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_mandatory_fields_missing. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_invariant_failure. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_initial_values_callable. Retrieved 1/4 statements.
# Partially parsed test_precord_new_with_initial_values_overridden. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields_and_ignore_extra. Retrieved 4/9 statements.
# Partially parsed test_precord_new_with_no_initial_and_no_special_attributes. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 42
    var_3 = 'test'

def test_case_0():
    var_0 = 'field1'
    var_1 = 21
    var_2 = set()

def test_case_0():
    var_0 = 'field1'
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'extra_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'test'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'TestRecord.field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'ERR1'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field1'
    var_2 = lambda : 100
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field1'
    var_2 = lambda : 100
    var_3 = {var_1: var_2}
    var_4 = 200

def test_case_0():
    var_0 = 'field1'
    var_1 = 5
    var_2 = 10
    var_3 = 'field1'
    var_4 = True
    var_5 = 'extra'

def test_case_0():
    var_0 = {}



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_new_without_special_attributes.
# Partially parsed test_new_with_regular_kwargs. Retrieved 2/5 statements.
# Partially parsed test_new_with_factory_fields. Retrieved 1/5 statements.
# Partially parsed test_new_with_ignore_extra. Retrieved 1/5 statements.
# Partially parsed test_new_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test_new_with_overridden_initial_values. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = 'x'
    var_1 = lambda : 10
    var_2 = {var_0: var_1}
    var_3 = 20

def test_case_0():
    var_0 = 'x'
    var_1 = lambda : 10
    var_2 = {var_0: var_1}
    var_3 = 30
    var_4 = 20



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_constructor_without_special_attributes. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_without_ignore_extra. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'value1'
    var_8 = 'value2'

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}
    var_3 = 0
    var_4 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = lambda : 'default1'
    var_9 = 'default2'
    var_10 = {var_6: var_8, var_7: var_9}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = lambda : 'default1'
    var_9 = 'default2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'custom1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'field1'
    var_8 = 'factory_value'
    var_9 = {var_7: var_8}
    var_10 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = True
    var_6 = 'value1'
    var_7 = 'extra'
    var_8 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'value1'
    var_6 = 'extra'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------

# Partially parsed test_precord_new_creates_instance_with_special_attributes. Retrieved 2/9 statements.
# Failed to parse test_precord_new_uses_evolver_for_normal_creation.
# Partially parsed test_precord_new_applies_initial_values. Retrieved 1/3 statements.
# Partially parsed test_precord_new_overrides_initial_values_with_kwargs. Retrieved 2/4 statements.
# Partially parsed test_precord_new_passes_factory_fields_to_evolver. Retrieved 1/5 statements.
# Partially parsed test_precord_new_passes_ignore_extra_to_evolver. Retrieved 1/5 statements.
# Partially parsed test_precord_new_handles_multiple_kwargs. Retrieved 2/5 statements.
# Partially parsed test_precord_new_raises_attribute_error_for_invalid_field. Retrieved 1/5 statements.
# Partially parsed test_precord_new_invokes_field_factories. Retrieved 13/19 statements.
# Partially parsed test_precord_new_validates_field_types. Retrieved 12/26 statements.
# Partially parsed test_precord_new_enforces_invariants. Retrieved 7/17 statements.
# Partially parsed test_precord_new_checks_mandatory_fields. Retrieved 1/4 statements.
# Failed to parse test_precord_new_checks_global_invariants.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = lambda : 10
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 7

def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 5
    var_1 = bool(False)
    assert var_1 is True
    var_2 = "'invalid_field' is not among the specified fields for TestRecord"

def test_case_0():
    var_0 = 'x'
    var_1 = 'Field'
    var_2 = ()
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = 2
    var_6 = lambda v: v * var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda v: var_9
    var_11 = {var_3: var_6, var_4: var_10}
    var_12 = [var_1, var_2, var_11]
    var_13 = 3

def test_case_0():
    var_0 = 'x'
    var_1 = 'Field'
    var_2 = ()
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda v: v
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda v: var_8
    var_10 = {var_3: var_5, var_4: var_9}
    var_11 = [var_1, var_2, var_10]
    var_12 = 'not_an_int'
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'Invalid type'

def test_case_0():
    var_0 = 'x'
    var_1 = 'Field'
    var_2 = ()
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda v: v
    var_6 = 5
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'INVARIANT_FAILED'

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestRecord.x'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 5/13 statements.
# Partially parsed test_set_with_invalid_type_raises_ptype_error. Retrieved 5/14 statements.
# Partially parsed test_set_with_failed_invariant_adds_error_code. Retrieved 5/15 statements.
# Partially parsed test_set_with_non_existent_field_raises_attribute_error. Retrieved 6/12 statements.
# Partially parsed test_set_with_factory_and_ignore_extra. Retrieved 7/17 statements.
# Partially parsed test_set_with_factory_invariant_exception_adds_errors. Retrieved 5/17 statements.
# Partially parsed test_set_with_factory_fields_skips_factory. Retrieved 6/16 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = 'Alice'

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'age'
    var_5 = 'not_an_int'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = ''
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'ERR_EMPTY'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'unknown'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = "'unknown' is not among the specified fields"

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = None
    var_5 = True
    var_6 = 'name'
    var_7 = 'alice'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = 'alice'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'FACTORY_ERR'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = set()
    var_5 = 'name'
    var_6 = 'alice'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_persistent_returns_instance_of_destination_class.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_failure. Retrieved 2/9 statements.
# Failed to parse test_persistent_raises_invariant_exception_on_missing_mandatory_fields.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 2/11 statements.
# Failed to parse test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type.
# Partially parsed test_persistent_constructs_new_instance_if_dirty. Retrieved 2/8 statements.
# Partially parsed test_persistent_aggregates_multiple_invariant_errors. Retrieved 4/13 statements.
# Partially parsed test_persistent_aggregates_missing_fields_from_set_and_mandatory. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'field'
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'field'
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'field'
    var_1 = 1

def test_case_0():
    var_0 = 'field1'
    var_1 = -1
    var_2 = 'field2'
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TestRecord.field1'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 5/13 statements.
# Partially parsed test_set_with_invalid_type_raises_ptype_error. Retrieved 5/13 statements.
# Partially parsed test_set_with_failed_invariant_adds_error_code. Retrieved 5/14 statements.
# Partially parsed test_set_with_non_existent_field_raises_attribute_error. Retrieved 6/12 statements.
# Partially parsed test_set_with_factory_field_and_ignore_extra. Retrieved 6/18 statements.
# Partially parsed test_set_with_factory_field_without_ignore_extra. Retrieved 5/17 statements.
# Partially parsed test_set_with_factory_field_invariant_exception_adds_errors. Retrieved 5/18 statements.
# Partially parsed test_set_with_non_factory_field_uses_original_value. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = 'Alice'

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'age'
    var_5 = 'not_an_int'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = ''
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'ERR_EMPTY'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'unknown'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'unknown'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = True
    var_6 = 'alice'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = 'alice'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'name'
    var_5 = 'alice'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'ERR_FACTORY'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = set()
    var_5 = 'name'
    var_6 = 'Alice'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_constructor_without_special_attributes. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_without_ignore_extra. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'value1'
    var_8 = 'value2'

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}
    var_3 = 0
    var_4 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = lambda : 'default1'
    var_9 = 'default2'
    var_10 = {var_6: var_8, var_7: var_9}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = lambda : 'default1'
    var_9 = 'default2'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 'custom1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'field1'
    var_8 = lambda x: x.upper()
    var_9 = {var_7: var_8}
    var_10 = 'test'
    var_11 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = True
    var_6 = 'value1'
    var_7 = 'extra'
    var_8 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = {}
    var_5 = 'value1'
    var_6 = 'extra'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_precord_initial_values_condition_true. Retrieved 2/4 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_errors. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_pmap_when_not_dirty_and_already_instance. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/15 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'mandatory_field'

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error_code'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'new_field'
    var_5 = 'new_value'



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 3/7 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_no_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 5/9 statements.
# Partially parsed test_precord_repr_with_special_characters_in_field_value. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = 'Alice'
    var_3 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = {}
    var_3 = 10
    var_4 = 'test'
    var_5 = "TestRecord(x=10, y='test')"

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'TestRecord()'

def test_case_0():
    var_0 = 'data'
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = "TestRecord(data={'key': 'value'})"

def test_case_0():
    var_0 = 'text'
    var_1 = {}
    var_2 = 'line1\nline2'
    var_3 = "TestRecord(text='line1\\nline2')"



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = ()
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = ()
    var_3 = 'field'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = 'new_field'
    var_4 = 'new_value'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_precord_repr_returns_correct_format. Retrieved 15/28 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'value1'
    var_8 = 42
    var_9 = 'TestRecord('
    var_10 = ')'
    var_11 = 'field1='
    var_12 = 'field2='
    var_13 = '42'
    var_14 = len(var_9)
    var_15 = len(var_10)
    var_16 = ', '
    var_17 = 'field1='
    var_18 = "'value1'"
    var_19 = '"value1"'
    var_20 = 'field2='
    var_21 = '42'



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_precord_repr_returns_correct_format. Retrieved 14/28 statements.


def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {}
    var_4 = 'value1'
    var_5 = 42
    var_6 = 'TestRecord('
    var_7 = ')'
    var_8 = 'field1='
    var_9 = 'field2='
    var_10 = '42'
    var_11 = len(var_6)
    var_12 = len(var_7)
    var_13 = ', '
    var_14 = 'field1='
    var_15 = "'value1'"
    var_16 = '"value1"'
    var_17 = 'field2='
    var_18 = '42'



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Partially parsed test_serialize_with_no_serializers. Retrieved 6/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_multiple_fields_and_mixed_serializers. Retrieved 9/15 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 5/11 statements.
# Partially parsed test_serialize_on_empty_record. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'value1'
    var_6 = 42
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = {var_7: var_5, var_8: var_6}

def test_case_0():
    var_0 = 'field'
    var_1 = 'data'
    var_2 = 'field'
    var_3 = 'custom_data'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = 3
    var_4 = 'hello'
    var_5 = 'world'
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = 'field3'
    var_9 = 6
    var_10 = 'WORLD'
    var_11 = {var_6: var_9, var_7: var_4, var_8: var_10}

def test_case_0():
    var_0 = 'field'
    var_1 = 'test'
    var_2 = 'json'
    var_3 = 'field'
    var_4 = 'json:test'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = {}
    var_1 = {}



# Parsed testcases at query #28
#--------------------------






####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_persistent_returns_instance_of_destination_class.
# Failed to parse test_persistent_raises_invariant_exception_on_missing_mandatory_fields.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_failure. Retrieved 1/9 statements.
# Failed to parse test_persistent_raises_invariant_exception_on_global_invariant_failure.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type. Retrieved 1/6 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 1/9 statements.
# Partially parsed test_persistent_creates_new_instance_if_pmap_not_of_destination_class. Retrieved 1/8 statements.
# Partially parsed test_persistent_aggregates_multiple_invariant_errors. Retrieved 2/13 statements.
# Partially parsed test_persistent_aggregates_missing_fields_and_invariant_errors. Retrieved 1/10 statements.


def test_case_0():
    var_0 = 0
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 0
    var_1 = 20
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 0
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_new_creates_instance_with_special_attributes. Retrieved 2/8 statements.
# Partially parsed test_precord_new_uses_evolver_for_initial_values. Retrieved 4/12 statements.
# Partially parsed test_precord_new_applies_initial_values_from_class. Retrieved 5/14 statements.
# Partially parsed test_precord_new_overrides_initial_values_with_kwargs. Retrieved 6/15 statements.
# Partially parsed test_precord_new_passes_factory_fields_to_evolver. Retrieved 1/7 statements.
# Partially parsed test_precord_new_passes_ignore_extra_to_evolver. Retrieved 1/7 statements.
# Partially parsed test_precord_new_raises_attribute_error_for_invalid_field. Retrieved 2/9 statements.
# Partially parsed test_precord_new_handles_invariant_exception. Retrieved 9/21 statements.
# Partially parsed test_precord_new_checks_mandatory_fields. Retrieved 3/12 statements.
# Partially parsed test_precord_new_validates_global_invariants. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = []
    var_3 = []
    var_4 = 10
    var_5 = lambda : var_4
    var_6 = 20

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = []
    var_3 = []
    var_4 = 10
    var_5 = lambda : var_4
    var_6 = 20
    var_7 = 30

def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = 'valid'
    var_1 = []
    var_2 = 5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = "'invalid' is not among the specified fields for TestRecord"

def test_case_0():
    var_0 = []
    var_1 = ()
    var_2 = ()
    var_3 = ()
    var_4 = ''
    var_5 = [var_4]
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = 'a'
    var_10 = 1
    var_11 = bool(False)
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = []
    var_3 = []
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = ()
    var_1 = ()
    var_2 = ()
    var_3 = ''
    var_4 = [var_3]
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 6/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 6/12 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 7/13 statements.
# Partially parsed test_serialize_empty_record. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'value1'
    var_6 = 42
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = {var_7: var_5, var_8: var_6}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'custom_value1'
    var_7 = {var_4: var_6, var_5: var_3}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = 'json'
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'json_value1'
    var_8 = {var_5: var_7, var_6: var_3}

def test_case_0():
    var_0 = {}
    var_1 = {}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = None
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = {var_6: var_5, var_7: var_5}



# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 3/15 statements.
# Partially parsed test_set_with_field_factory_exception. Retrieved 3/19 statements.
# Partially parsed test_set_with_field_factory_invariant_exception. Retrieved 3/17 statements.
# Partially parsed test_set_with_ignore_extra_complaint. Retrieved 4/20 statements.
# Partially parsed test_set_with_type_check_failure. Retrieved 3/16 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 3/15 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 4/11 statements.
# Partially parsed test_set_with_factory_fields_skipping_factory. Retrieved 9/26 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = True
    var_3 = 'key'
    var_4 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'nonexistent'
    var_3 = 5
    var_4 = bool(False)
    assert var_4 is True
    var_5 = "'nonexistent' is not among the specified fields"

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'key'
    var_6 = {}
    var_7 = set()
    var_8 = 'key'
    var_9 = 5
    var_10 = bool(var_0 == [])
    assert var_10 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 6/13 statements.
# Partially parsed test_serialize_without_custom_serializer. Retrieved 5/11 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 5/10 statements.
# Partially parsed test_serialize_empty_record. Retrieved 1/6 statements.
# Partially parsed test_serialize_multiple_fields_mixed_serializers. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 'name'
    var_3 = 'value'
    var_4 = 10
    var_5 = {var_2: var_0, var_3: var_4}

def test_case_0():
    var_0 = 'test'
    var_1 = 5
    var_2 = 'name'
    var_3 = 'value'
    var_4 = {var_2: var_0, var_3: var_1}

def test_case_0():
    var_0 = 'info'
    var_1 = 'json'
    var_2 = 'data'
    var_3 = 'json:info'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = 3.5
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 11
    var_7 = 7.0
    var_8 = {var_3: var_6, var_4: var_1, var_5: var_7}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 8/12 statements.
# Partially parsed test_serialize_without_custom_serializer. Retrieved 7/11 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_none_format. Retrieved 6/10 statements.
# Partially parsed test_serialize_empty_record. Retrieved 1/6 statements.
# Partially parsed test_serialize_multiple_fields_mixed_serializers. Retrieved 11/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda fmt, v: f'serialized_{v}'
    var_2 = module_0.field(serializer=var_1)
    var_3 = 'test'
    var_4 = 42
    var_5 = 'name'
    var_6 = 'value'
    var_7 = 'serialized_42'
    var_8 = {var_5: var_3, var_6: var_7}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42
    var_4 = 'name'
    var_5 = 'value'
    var_6 = {var_4: var_2, var_5: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, v: f'{fmt}_{v}'
    var_1 = module_0.field(serializer=var_0)
    var_2 = 100
    var_3 = 'fmt'
    var_4 = 'value'
    var_5 = 'fmt_100'
    var_6 = {var_4: var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, v: f'serialized_{v}'
    var_1 = module_0.field(serializer=var_0)
    var_2 = 200
    var_3 = None
    var_4 = 'value'
    var_5 = 'serialized_200'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = {}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda fmt, v: v * 2
    var_2 = module_0.field(serializer=var_1)
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = 4
    var_11 = {var_7: var_4, var_8: var_10, var_9: var_6}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_precord_initial_values_condition_true. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = lambda : 42
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'field2'
    var_8 = 'custom'
    var_9 = {var_7: var_8}



# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 9/89 statements.
# Partially parsed test___new___handles_no_invariants. Retrieved 3/62 statements.
# Failed to parse test___new___raises_on_non_callable_invariant.


def test_case_0():
    var_0 = []
    var_1 = 'base_field'
    var_2 = True
    var_3 = (var_2,)
    var_4 = 'another_field'
    var_5 = 'custom_field'
    var_6 = False
    var_7 = 'error'
    var_8 = (var_6, var_7)
    var_9 = 'TestClass'
    var_10 = '_precord_fields'
    var_11 = 'base_field'
    var_12 = 'another_field'
    var_13 = 'custom_field'
    var_14 = '_precord_invariants'
    var_15 = '_precord_mandatory_fields'
    var_16 = 'another_field'
    var_17 = 'base_field'
    var_18 = 'custom_field'
    var_19 = '_precord_initial_values'
    var_20 = '__slots__'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'TestClass'
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '_precord_mandatory_fields'
    var_6 = set()
    var_7 = '_precord_initial_values'
    var_8 = '__slots__'



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.
# Partially parsed test_persistent_aggregates_multiple_invariant_errors. Retrieved 7/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'mandatory_field'

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error_code'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'mandatory'
    var_3 = {var_2}
    var_4 = []
    var_5 = {}
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = 'field2'
    var_9 = 'value2'
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'error1'
    var_12 = 'error2'
    var_13 = 'mandatory'



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 6/12 statements.
# Partially parsed test_serialize_without_custom_serializer. Retrieved 5/12 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 4/9 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_mixed_serializers. Retrieved 8/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda fmt, x: f'Age: {x}'
    var_2 = module_0.field(serializer=var_1)
    var_3 = 'Alice'
    var_4 = 30
    var_5 = None
    var_6 = 'Age: 30'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Bob'
    var_3 = 100
    var_4 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, x: f'{fmt}: {x}'
    var_1 = module_0.field(serializer=var_0)
    var_2 = 'test'
    var_3 = 'json'
    var_4 = 'json: test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda fmt, x: x * 2
    var_2 = module_0.field(serializer=var_1)
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = None
    var_8 = 4



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 5/9 statements.
# Partially parsed test_precord_repr_with_no_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 7/11 statements.
# Partially parsed test_precord_repr_with_special_characters_in_field_value. Retrieved 4/8 statements.


def test_case_0():
    var_0 = ()
    var_1 = 'name'
    var_2 = {}
    var_3 = 'Alice'
    var_4 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = {}
    var_4 = 10
    var_5 = 'test'
    var_6 = "TestRecord(x=10, y='test')"

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}
    var_3 = 'TestRecord()'

def test_case_0():
    var_0 = ()
    var_1 = 'data'
    var_2 = 'count'
    var_3 = {}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 5
    var_8 = "TestRecord(data={'key': 'value'}, count=5)"

def test_case_0():
    var_0 = ()
    var_1 = 'text'
    var_2 = {}
    var_3 = 'line1\nline2'
    var_4 = "TestRecord(text='line1\\nline2')"



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_new_with_kwargs_overrides_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 2/5 statements.
# Partially parsed test_precord_new_without_ignore_extra_raises_attribute_error. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_mandatory_fields_missing_raises_invariant_exception. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_field_invariant_failure_raises_invariant_exception. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_global_invariant_failure_raises_invariant_exception. Retrieved 3/7 statements.
# Partially parsed test_precord_new_with_valid_data_creates_record. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 20

def test_case_0():
    var_0 = 'field1'
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = 5

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 2
    var_3 = 'extra_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = "'extra_field' is not among the specified fields for TestRecord"

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = {var_2}
    var_4 = 'test'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'TestRecord.field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = -5
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'ERR_POSITIVE'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = lambda r: (r['field1'] + r['field2'] > 0, 'ERR_SUM_POSITIVE')
    var_3 = [var_2]
    var_4 = -10
    var_5 = 5
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'ERR_SUM_POSITIVE'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 42
    var_3 = 'answer'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_class. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'mandatory_field'

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/15 statements.
# Partially parsed test_precord_constructor_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty_record. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_nested_initial_values. Retrieved 6/9 statements.


def test_case_0():
    var_0 = ()
    var_1 = 0
    var_2 = []

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'default'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = ()
    var_1 = 'name'
    var_2 = 'value'
    var_3 = 'test'
    var_4 = 10

def test_case_0():
    var_0 = ()
    var_1 = 'items'
    var_2 = 'items'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = ()
    var_1 = 'valid'
    var_2 = True
    var_3 = 'yes'
    var_4 = 'no'
    var_5 = 'extra'

def test_case_0():
    var_0 = ()
    var_1 = 'counter'
    var_2 = 'counter'
    var_3 = lambda : 100
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = ()
    var_1 = 'data'
    var_2 = 'data'
    var_3 = 'initial'
    var_4 = {var_2: var_3}
    var_5 = 'overridden'

def test_case_0():
    var_0 = ()
    var_1 = {}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'nested'
    var_2 = 'nested'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_2: var_6}
    var_8 = 'key'
    var_9 = 'value'
    var_10 = {var_8: var_9}
    var_11 = module_0.pmap(var_10)



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_kwargs_overrides. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_without_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_empty_record. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_all_kwargs. Retrieved 3/6 statements.


def test_case_0():
    var_0 = ()
    var_1 = 0
    var_2 = ()

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'default'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'default'
    var_6 = 42
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'overridden'

def test_case_0():
    var_0 = ()
    var_1 = 'field'
    var_2 = True

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'value1'
    var_3 = 'extra'
    var_4 = True
    var_5 = 'extra_field'

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'value1'
    var_3 = 'extra'
    var_4 = False
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = ()
    var_1 = 'field'
    var_2 = 'field'
    var_3 = lambda : 999
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = ()
    var_1 = {}

def test_case_0():
    var_0 = ()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'hello'
    var_4 = 123



# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.
# Partially parsed test_persistent_creates_new_instance_if_pm_not_instance_of_cls. Retrieved 4/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'mandatory_field'

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = set()



