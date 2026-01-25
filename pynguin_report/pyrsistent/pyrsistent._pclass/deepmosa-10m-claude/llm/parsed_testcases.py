####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_set_updates_data_and_marks_dirty. Retrieved 5/10 statements.
# Partially parsed test_set_with_same_value_does_not_mark_dirty. Retrieved 3/10 statements.
# Partially parsed test_set_overwrites_existing_value. Retrieved 4/9 statements.
# Partially parsed test_set_returns_self_for_chaining. Retrieved 5/11 statements.
# Partially parsed test_set_with_none_value. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = 'key2'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'key1'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'old_value'
    var_2 = {var_0: var_1}
    var_3 = 'new_value'
    var_4 = 'key1'

def test_case_0():
    var_0 = {}
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = 'key2'
    var_4 = 'value2'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = None
    var_5 = 'key2'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_type_check. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Failed to parse test_pclass_new_empty.
# Partially parsed test_pclass_new_multiple_invariant_errors. Retrieved 2/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

def test_case_0():
    var_0 = 'string'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -5
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = -1
    var_1 = -2
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 8/12 statements.
# Partially parsed test_remove_nonexistent_item. Retrieved 7/12 statements.
# Partially parsed test_remove_item_that_was_set. Retrieved 8/13 statements.
# Partially parsed test_remove_marks_data_as_dirty. Retrieved 6/10 statements.
# Partially parsed test_remove_multiple_items. Retrieved 10/15 statements.


def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'value1'
    var_7 = 'value2'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'key1'
    var_10 = 'key1'

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = 'nonexistent'
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}
    var_7 = 'key2'
    var_8 = 'value2'
    var_9 = 'key2'
    var_10 = 'key2'

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key1'
    var_5 = 'value1'
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'key1'
    var_5 = 'key2'
    var_6 = 'key3'
    var_7 = 'value1'
    var_8 = 'value2'
    var_9 = 'value3'
    var_10 = {var_4: var_7, var_5: var_8, var_6: var_9}
    var_11 = 'key1'
    var_12 = 'key3'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_preserves_other_fields. Retrieved 7/11 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_with_initial_field. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 99
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 5



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_field_invariant_errors. Retrieved 1/9 statements.
# Partially parsed test_pclass_raises_invariant_exception_with_both_errors_and_missing_fields. Retrieved 2/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'TestClass.y'

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_optional_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 2/11 statements.
# Partially parsed test_serialize_with_nested_objects. Retrieved 3/9 statements.
# Partially parsed test_serialize_multiple_fields_with_mixed_serializers. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 42
    var_4 = 'x'

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 123
    var_1 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = None
    var_4 = 'present'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_partial_fields. Retrieved 6/10 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 2/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)
    var_2 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_remove_item_exists_in_data. Retrieved 8/12 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'MockPClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]
    var_9 = 'key1'
    var_10 = 'key1'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 5/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_fields'
    var_5 = 'x'
    var_6 = 'y'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_meta_new_creates_class_with_fields. Retrieved 5/13 statements.
# Partially parsed test_pclass_meta_new_sets_slots. Retrieved 3/8 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_slot. Retrieved 2/6 statements.
# Partially parsed test_pclass_meta_new_stores_invariants. Retrieved 3/13 statements.
# Partially parsed test_pclass_meta_new_inherits_fields_from_base. Retrieved 6/15 statements.
# Partially parsed test_pclass_meta_new_field_removed_from_dict. Retrieved 4/12 statements.
# Partially parsed test_pclass_meta_new_multiple_fields. Retrieved 5/15 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'test_attr'
    var_3 = 'TestClass'
    var_4 = '_pclass_fields'
    var_5 = 'test_attr'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__slots__'
    var_3 = '_pclass_frozen'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = '_pclass_invariants'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'base_field'
    var_3 = 'BaseClass'
    var_4 = {}
    var_5 = 'DerivedClass'
    var_6 = 'base_field'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'test_attr'
    var_3 = 'TestClass'
    var_4 = 'test_attr'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'TestClass'
    var_5 = 'field1'
    var_6 = 'field2'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_without_factory_fields. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/9 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

def test_case_0():
    var_0 = 'string'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type'

def test_case_0():
    var_0 = '5'
    var_1 = 'x'
    var_2 = {var_1}

def test_case_0():
    var_0 = 5
    var_1 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Value must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'x must be greater than y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 30
    var_3 = module_0.field(initial=var_2)
    var_4 = 1
    var_5 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra_true. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_field_invariant_violation. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant_violation. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_creates_independent_instances. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_without_factory_fields_parameter. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '5'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -5
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum must be positive'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : []
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '5'
    var_2 = 10
    var_3 = 'x'
    var_4 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pclass_hash_same_values_same_hash. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values_different_hash. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_hashable_in_set. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_hashable_as_dict_key. Retrieved 5/10 statements.
# Partially parsed test_pclass_hash_with_nested_values. Retrieved 7/13 statements.
# Partially parsed test_pclass_hash_with_string_fields. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_consistent_across_calls. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'value1'
    var_4 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_3, var_4)
    var_6 = (var_3, var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 'data'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_frozen'



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_is_pclass_with_single_checked_type_base.
# Failed to parse test_is_pclass_with_multiple_bases.
# Failed to parse test_is_pclass_with_different_single_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False

import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 1/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

def test_case_0():
    var_0 = '5'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_partial_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_no_fields. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_multiple_types. Retrieved 9/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'string'
    var_4 = 42
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pclass_meta_new_is_pclass_predicate_false. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = '__weakref__'
    var_4 = '_pclass_frozen'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_single_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_complex_values. Retrieved 9/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5
    var_4 = 'x'
    var_5 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = 'b'
    var_8 = {var_6: var_2, var_7: var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_predicate_line_25_with_invariant_errors. Retrieved 2/10 statements.
# Partially parsed test_pclass_predicate_line_25_with_missing_fields. Retrieved 1/6 statements.
# Partially parsed test_pclass_predicate_line_25_both_conditions. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = -1
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_hash. Retrieved 4/11 statements.
# Partially parsed test_pclass_hash_different_for_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_works_with_missing_values. Retrieved 3/8 statements.
# Partially parsed test_pclass_hash_is_hashable. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_with_complex_values. Retrieved 8/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'hello'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = (var_3, var_4, var_5)
    var_7 = (var_3, var_4, var_5)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_frozen'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/13 statements.
# Partially parsed test_pclass_hash_different_for_different_values. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_with_optional_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_hash_allows_use_in_set. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_allows_use_as_dict_key. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'value1'
    var_4 = 'value2'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pclass_eq_same_class_same_values. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_same_class_different_values. Retrieved 5/8 statements.
# Partially parsed test_pclass_eq_different_class. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_with_non_pclass_object. Retrieved 4/6 statements.
# Partially parsed test_pclass_eq_with_missing_values. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_one_has_missing_value_other_doesnt. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_both_have_missing_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_eq_reflexive. Retrieved 2/4 statements.
# Partially parsed test_pclass_eq_with_none_values. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_complex_values. Retrieved 10/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2: var_1}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

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
    var_8 = [var_2, var_3, var_4]
    var_9 = {var_6: var_2}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_multiple_fields_with_mixed_types. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 5
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '5'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = module_0.field(initial=var_1)
    var_3 = True
    var_4 = module_0.field(mandatory=var_3)
    var_5 = 'hello'
    var_6 = 3.14



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 5/8 statements.
# Partially parsed test_remove_nonexistent_item. Retrieved 4/8 statements.
# Partially parsed test_remove_multiple_items. Retrieved 7/11 statements.
# Partially parsed test_remove_discards_from_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_remove_sets_dirty_flag. Retrieved 3/6 statements.
# Partially parsed test_delitem_calls_remove. Retrieved 3/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'key1'
    var_7 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'nonexistent'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'key3'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 'key1'
    var_9 = 'key2'
    var_10 = 'key3'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key1'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 6/10 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_fields_subset. Retrieved 9/13 statements.
# Partially parsed test_pclass_new_empty. Retrieved 1/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 1
    var_1 = 'hello'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 2
    var_7 = {var_3: var_4}
    var_8 = module_1.pmap(var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 8/24 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_multiple_valid_types. Retrieved 7/20 statements.
# Partially parsed test_check_and_set_attr_no_type_constraint. Retrieved 6/20 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_0 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 'not_an_int'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'test_field'
    var_9 = bool(var_0 == [])
    assert var_9 is True

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'value_too_small'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_0 == ['value_too_small'])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 'hello'
    var_7 = bool(var_0 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True
    var_3 = (var_2, var_1)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = []
    var_7 = bool(var_0 == [])
    assert var_7 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_reduce_returns_tuple_with_restore_pickle_and_class_data. Retrieved 4/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_is_pclass_returns_false_for_empty_bases. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '_pclass_fields'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = '__weakref__'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_none_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 2



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 5/10 statements.
# Partially parsed test_pclass_repr_string_field. Retrieved 2/6 statements.
# Failed to parse test_pclass_repr_empty_pclass.
# Partially parsed test_pclass_repr_nested_pclass. Retrieved 3/9 statements.
# Partially parsed test_pclass_repr_with_list_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_with_optional_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_repr_with_special_characters. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'MultiFieldClass('
    var_5 = 'x=1'
    var_6 = 'y=2'
    var_7 = ')'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'OuterClass('
    var_4 = 'InnerClass(value=42)'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'ListClass('
    var_6 = 'items=[1, 2, 3]'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 'value'
    var_4 = 'OptionalClass('
    var_5 = "required='value'"
    var_6 = 'optional=None'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello\nworld'
    var_2 = 'SpecialClass('
    var_3 = 'text='
    var_4 = 'hello'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_hash_returns_integer. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raise_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_fields_with_mixed_initial. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 3



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/19 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_attr'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'error_code_1'



# Parsed testcases at query #36
#--------------------------

# Failed to parse test_pclass_meta_new_with_pclass_bases.




# Parsed testcases at query #37
#--------------------------

# Partially parsed test_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_repr_with_multiple_fields. Retrieved 7/12 statements.
# Partially parsed test_repr_with_string_field. Retrieved 2/6 statements.
# Partially parsed test_repr_with_nested_objects. Retrieved 3/9 statements.
# Partially parsed test_repr_with_missing_optional_field. Retrieved 4/8 statements.
# Partially parsed test_repr_with_list_field. Retrieved 5/9 statements.
# Partially parsed test_repr_with_dict_field. Retrieved 4/8 statements.
# Failed to parse test_repr_empty_class.
# Partially parsed test_repr_with_boolean_field. Retrieved 2/6 statements.
# Partially parsed test_repr_with_float_field. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'MultiFieldClass('
    var_7 = 'x=1'
    var_8 = 'y=2'
    var_9 = 'z=3'
    var_10 = ')'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'OuterClass('
    var_4 = 'InnerClass(value=42)'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = None
    var_5 = 'OptionalFieldClass('
    var_6 = 'x=1'
    var_7 = 'y=None'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'ListClass('
    var_6 = 'items=[1, 2, 3]'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'DictClass('
    var_5 = 'data='
    var_6 = "'key': 'value'"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 3.14



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots. Retrieved 4/18 statements.
# Partially parsed test_pclass_meta_new_sets_pclass_fields. Retrieved 3/16 statements.
# Partially parsed test_pclass_meta_new_sets_pclass_invariants. Retrieved 4/2 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_for_direct_subclass. Retrieved 2/6 statements.
# Partially parsed test_pclass_meta_new_no_weakref_for_indirect_subclass. Retrieved 2/8 statements.
# Partially parsed test_pclass_meta_new_inherits_fields. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'TestPClass'
    var_3 = '__slots__'
    var_4 = '_pclass_frozen'
    var_5 = 'field1'
    var_6 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestPClass'
    var_2 = '_pclass_fields'
    var_3 = 'field1'

def test_case_0():
    var_0 = True
    var_1 = '__invariant__'
    var_2 = 'TestPClass'
    var_3 = '_pclass_invariants'

def test_case_0():
    var_0 = True
    var_1 = '__invariant__'
    var_2 = 'TestPClass'
    var_3 = '_pclass_invariants'

def test_case_0():
    var_0 = {}
    var_1 = 'TestPClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = {}
    var_1 = 'SecondPClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = 'child_field'
    var_2 = 'ChildPClass'
    var_3 = 'parent_field'
    var_4 = 'child_field'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_fields_with_defaults. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)
    var_4 = 3
    var_5 = module_0.field(initial=var_4)



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_is_pclass_returns_false_for_empty_bases. Retrieved 3/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'TestClass'
    var_3 = [var_2, var_1, var_0]
    var_4 = '__weakref__'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_repr_returns_formatted_string. Retrieved 6/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestClass('
    var_5 = ')'
    var_6 = 'x=1'
    var_7 = "y='hello'"



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_repr_returns_correct_format. Retrieved 6/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestClass'
    var_5 = 'x=1'
    var_6 = "y='hello'"
    var_7 = 'TestClass('
    var_8 = ')'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pclass_meta_weakref_not_added_when_not_pclass. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 0
    var_2 = None
    var_3 = 'TestClass'
    var_4 = '__weakref__'
    var_5 = '_pclass_frozen'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 10/18 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x'
    var_7 = 'y'
    var_8 = 'z'
    var_9 = {var_6, var_7, var_8}



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots_for_fields. Retrieved 5/14 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_slot_for_base_pclass. Retrieved 3/10 statements.
# Partially parsed test_pclass_meta_new_moves_fields_to_pclass_fields. Retrieved 3/10 statements.
# Partially parsed test_pclass_meta_new_stores_invariants. Retrieved 2/13 statements.
# Partially parsed test_pclass_meta_new_inherits_fields_from_bases. Retrieved 6/20 statements.
# Partially parsed test_pclass_meta_new_no_weakref_for_non_base_pclass. Retrieved 5/16 statements.


def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'TestClass'
    var_4 = '__slots__'
    var_5 = '_pclass_frozen'
    var_6 = 'field1'
    var_7 = 'field2'

def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'BaseClass'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'TestClass'
    var_3 = '_pclass_fields'
    var_4 = 'field1'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = '_pclass_invariants'

def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'Parent'
    var_3 = 'field2'
    var_4 = 'Child'
    var_5 = '_pclass_fields'
    var_6 = 1
    var_7 = 'field2'

def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'Parent'
    var_3 = 'field2'
    var_4 = 'Child'
    var_5 = '__weakref__'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_pclass_new_basic_initialization. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_extra_fields_raise_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/12 statements.
# Failed to parse test_pclass_new_empty_class.
# Partially parsed test_pclass_new_all_fields_with_initial. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Global invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_set_method_predicate_at_line_25. Retrieved 6/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10
    var_5 = '_pclass_fields'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_invariant_errors_present. Retrieved 4/11 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_invariant_exception_with_both_errors_and_missing. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '_check_and_set_attr'
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    var_4 = 'RequiredClass.required_field'
    var_5 = bool(var_3)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_pclass_new_basic_field_assignment. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_extra_kwargs_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_cannot_modify_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_field_invariant_violation. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra_true. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_new_with_type_check_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_global_invariant_violation. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_all_fields_with_initial. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_mixed_initial_and_mandatory. Retrieved 3/6 statements.
# Failed to parse test_pclass_new_empty_class.
# Partially parsed test_pclass_new_with_factory_and_type. Retrieved 1/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 5
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

def test_case_0():
    var_0 = 1
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Global invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = 20

def test_case_0():
    var_0 = '42'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : []
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_args. Retrieved 4/8 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_with_optional_field. Retrieved 4/8 statements.
# Partially parsed test_set_preserves_unmodified_fields. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 20



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_set_method_iterates_over_pclass_fields. Retrieved 8/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = '_pclass_fields'
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 3/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'test_value'
    var_3 = bool(var_0 == ['invariant_error_code'])
    assert var_3 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_value'
    var_2 = []
    var_3 = bool(var_2 == ['invariant_error_code'])
    assert var_3 is True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

def test_case_0():
    var_0 = '5'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_cannot_set_after_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/11 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/14 statements.
# Partially parsed test_pclass_new_partial_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 8/12 statements.
# Partially parsed test_pclass_new_multiple_invariant_errors. Retrieved 2/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

def test_case_0():
    var_0 = 'string'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must_be_positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum_too_small'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 20
    var_2 = module_0.field(initial=var_1)
    var_3 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = {var_3}
    var_5 = True
    var_6 = {var_0: var_5}
    var_7 = module_0.pmap(var_6)

def test_case_0():
    var_0 = -1
    var_1 = -2
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 5/21 statements.


def test_case_0():
    var_0 = []
    var_1 = 'value'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'invariant_error_code'
    var_5 = None



# Parsed testcases at query #62
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_preserves_values. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_nested_structures. Retrieved 4/8 statements.
# Partially parsed test_serialize_returns_dict. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 42
    var_4 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'Alice'
    var_4 = 30
    var_5 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 2/5 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_multiple_instances_independent. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_multiple_invariant_errors. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_no_arguments. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : []
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'frozen'
    var_4 = bool('frozen' in str(e).lower())
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/9 statements.
# Partially parsed test_pclass_constructor_no_arguments. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_multiple_missing_mandatory_fields. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 5
    var_3 = {var_1: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 1/5 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_freezes_object. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_multiple_fields_with_mixed_initial. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'are not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 1
    var_6 = 3



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_no_factory_fields. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_all_optional_fields. Retrieved 4/9 statements.
# Partially parsed test_pclass_constructor_multiple_missing_mandatory_fields. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_field_invariant_error. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '5'
    var_1 = 'x'
    var_2 = {var_1}

def test_case_0():
    var_0 = '5'
    var_1 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v > 0
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 1/5 statements.
# Failed to parse test_pclass_constructor_empty.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'y'

def test_case_0():
    var_0 = '42'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 7/12 statements.
# Partially parsed test_pclass_repr_with_string_values. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_nested_object. Retrieved 3/9 statements.
# Failed to parse test_pclass_repr_empty_class.
# Partially parsed test_pclass_repr_with_list_value. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_with_none_value. Retrieved 1/5 statements.
# Partially parsed test_pclass_repr_with_boolean_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_with_float_value. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'MultiFieldClass('
    var_7 = 'x=1'
    var_8 = 'y=2'
    var_9 = 'z=3'
    var_10 = ')'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'OuterClass('
    var_4 = 'InnerClass(value=42)'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = True
    var_3 = False
    var_4 = 'BoolClass('
    var_5 = 'flag1=True'
    var_6 = 'flag2=False'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 3.14



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_pclass_invariant_errors_with_invariant_check. Retrieved 2/9 statements.
# Partially parsed test_pclass_predicate_line_25_true_with_missing_field. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 6/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_reduce_returns_restore_pickle_and_class_data. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_preserves_other_fields. Retrieved 7/11 statements.
# Partially parsed test_set_with_args_and_kwargs. Retrieved 7/11 statements.
# Partially parsed test_set_creates_immutable_copy. Retrieved 3/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 10
    var_6 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #81
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_pclass_hash_returns_integer. Retrieved 4/9 statements.
# Partially parsed test_pclass_hash_consistency. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_hash_usable_in_set. Retrieved 5/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_pclass_meta_new_creates_class_with_fields_and_invariants. Retrieved 4/12 statements.
# Partially parsed test_pclass_meta_new_sets_slots. Retrieved 3/10 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_for_top_level_class. Retrieved 2/5 statements.
# Partially parsed test_pclass_meta_new_without_weakref_for_subclass. Retrieved 4/12 statements.
# Partially parsed test_pclass_meta_new_with_invariant. Retrieved 4/15 statements.
# Partially parsed test_pclass_meta_new_removes_fields_from_dct. Retrieved 4/12 statements.


def test_case_0():
    var_0 = 'test_attr'
    var_1 = 'TestClass'
    var_2 = '_pclass_fields'
    var_3 = 'test_attr'
    var_4 = '_pclass_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestClass'
    var_2 = '__slots__'
    var_3 = '_pclass_frozen'
    var_4 = 'field1'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = {}
    var_1 = 'ParentClass'
    var_2 = 'field1'
    var_3 = 'ChildClass'
    var_4 = '__weakref__'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = '_pclass_invariants'
    var_3 = 0

def test_case_0():
    var_0 = 'my_field'
    var_1 = 'other_attr'
    var_2 = 'value'
    var_3 = 'TestClass'
    var_4 = 'my_field'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_pclass_predicate_line_25_with_invariant_errors. Retrieved 2/10 statements.
# Partially parsed test_pclass_predicate_line_25_with_missing_fields. Retrieved 1/6 statements.
# Partially parsed test_pclass_predicate_line_25_with_both_errors_and_missing. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'invariant_error'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 5/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_fields'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_eq_predicate_isinstance_true. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'invariant_error'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_all_mandatory_fields_provided. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_partial_mandatory_fields_missing. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = 1
    var_5 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'TestClass.y'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_becomes_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_no_fields.
# Partially parsed test_pclass_constructor_with_multiple_initial_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = 'inner'
    var_6 = {var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 3



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
# Failed to parse test_pclass_constructor_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_pclass_meta_new_predicate_line_1. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_create_from_same_class. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_multiple_fields_with_mixed_initial. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '5'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 3
    var_4 = module_0.field(initial=var_3)
    var_5 = 2



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_pclass_hash_returns_integer. Retrieved 4/9 statements.
# Partially parsed test_pclass_hash_consistency. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_single_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_hash_with_optional_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_hash_usable_in_dict. Retrieved 4/11 statements.
# Partially parsed test_pclass_hash_usable_in_set. Retrieved 3/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'first'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Failed to parse test_pclass_reduce_with_no_fields.
# Partially parsed test_pclass_reduce_with_complex_values. Retrieved 10/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'a'
    var_8 = {var_7: var_3}
    var_9 = 'string'



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 5/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_is_pclass_returns_false_for_empty_bases. Retrieved 3/5 statements.


def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'TestClass'
    var_3 = '__weakref__'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_serialize_with_no_fields. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_nested_pclass. Retrieved 5/11 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_all_field_types. Retrieved 13/17 statements.
# Partially parsed test_serialize_returns_dict. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'hello'
    var_5 = 3.14

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 42
    var_4 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test_data'
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 123
    var_5 = 'text'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_with_invariant_errors. Retrieved 2/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 6/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_repr_format. Retrieved 6/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestClass('
    var_5 = ')'
    var_6 = 'x=1'
    var_7 = "y='hello'"



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 8/22 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 7/19 statements.
# Partially parsed test_check_and_set_attr_multiple_valid_types. Retrieved 7/20 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_0 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = True
    var_3 = None
    var_4 = (var_2, var_3)
    var_5 = lambda x: var_4
    var_6 = 'test_field'
    var_7 = 'invalid'
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = []
    var_1 = 'invalid_value'
    var_2 = False
    var_3 = (var_2, var_1)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_0 == [var_1])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True
    var_3 = (var_2, var_1)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 'any_value'
    var_7 = bool(var_0 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_field'
    var_6 = 'hello'
    var_7 = bool(var_0 == [])
    assert var_7 is True



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_field_with_factory. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

def test_case_0():
    var_0 = '42'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 2/7 statements.
# Partially parsed test_pclass_missing_mandatory_field_raises_exception. Retrieved 1/6 statements.
# Partially parsed test_pclass_both_invariant_errors_and_missing_fields. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: (False, 'test error')
    var_1 = module_0.field(invariant=var_0)
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: (False, 'x error')
    var_2 = module_0.field(invariant=var_1, mandatory=var_0)
    var_3 = True
    var_4 = module_0.field(mandatory=var_3)
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'x error'
    var_8 = 'TestClass.y'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_set_method_iterates_over_pclass_fields. Retrieved 8/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = '_pclass_fields'
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 4/9 statements.
# Partially parsed test_pclass_constructor_with_invariant_error. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 20
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'TestClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'x must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -5
    var_3 = bool(False)
    assert var_3 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 8/12 statements.
# Partially parsed test_remove_item_not_in_data. Retrieved 7/12 statements.
# Partially parsed test_remove_multiple_items. Retrieved 10/17 statements.
# Partially parsed test_remove_using_delitem. Retrieved 8/13 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'MockPClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]
    var_9 = 'key1'
    var_10 = 'key1'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'MockPClass'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = 'nonexistent_key'
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'key3'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = 'MockPClass'
    var_8 = ()
    var_9 = {}
    var_10 = [var_7, var_8, var_9]
    var_11 = 'key1'
    var_12 = 'key2'
    var_13 = 'key3'
    var_14 = 'key1'
    var_15 = 'key2'

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'MockPClass'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]
    var_9 = 'key1'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_preserves_original. Retrieved 5/9 statements.
# Partially parsed test_set_with_single_field. Retrieved 3/7 statements.
# Partially parsed test_set_returns_same_class_type. Retrieved 3/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 99

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_method_iterates_over_pclass_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_eq_same_class_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_same_class_different_values. Retrieved 5/9 statements.
# Partially parsed test_pclass_eq_different_class. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_non_pclass. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_missing_values. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_one_missing_one_present. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_reflexive. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_symmetric. Retrieved 2/6 statements.
# Partially parsed test_pclass_eq_transitive. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2: var_1}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_instances_independent. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_nested_pclass. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'frozen'
    var_4 = bool('frozen' in str(e).lower())
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_hash_same_values_same_hash. Retrieved 4/10 statements.
# Partially parsed test_hash_different_values_different_hash. Retrieved 5/11 statements.
# Partially parsed test_hash_missing_field_values. Retrieved 3/9 statements.
# Partially parsed test_hash_usable_in_set. Retrieved 3/10 statements.
# Partially parsed test_hash_usable_in_dict. Retrieved 5/10 statements.
# Partially parsed test_hash_with_nested_values. Retrieved 7/13 statements.
# Partially parsed test_hash_consistent_across_calls. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'value1'
    var_4 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = (var_2, var_3, var_4)
    var_6 = (var_2, var_3, var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_partial_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_multiple_fields. Retrieved 12/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 'test'
    var_6 = 2
    var_7 = 3
    var_8 = [var_4, var_6, var_7]
    var_9 = 'key'
    var_10 = 'value'
    var_11 = {var_9: var_10}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/11 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_ignore_extra_true. Retrieved 7/10 statements.
# Partially parsed test_pclass_new_with_partial_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_set_attribute_on_frozen. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_multiple_instances_independent. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_factory_fields_parameter. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 1
    var_1 = 'hello'

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = '_pclass_frozen'
    var_3 = False

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Failed to parse test_pclass_reduce_with_no_fields.
# Partially parsed test_pclass_reduce_with_complex_values. Retrieved 9/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots. Retrieved 7/16 statements.
# Partially parsed test_pclass_meta_new_sets_pclass_fields. Retrieved 5/13 statements.
# Partially parsed test_pclass_meta_new_sets_pclass_invariants. Retrieved 3/16 statements.
# Partially parsed test_pclass_meta_new_removes_field_from_dct. Retrieved 4/15 statements.
# Partially parsed test_pclass_meta_new_without_weakref_for_subclass. Retrieved 6/15 statements.
# Partially parsed test_pclass_meta_new_inherits_fields_from_bases. Retrieved 7/18 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 1
    var_3 = None
    var_4 = 2
    var_5 = 'TestClass'
    var_6 = '__slots__'
    var_7 = '_pclass_frozen'
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = '__weakref__'

def test_case_0():
    var_0 = 'field1'
    var_1 = 1
    var_2 = None
    var_3 = 'TestClass'
    var_4 = '_pclass_fields'
    var_5 = 'field1'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = '_pclass_invariants'

def test_case_0():
    var_0 = 1
    var_1 = None
    var_2 = 'field1'
    var_3 = 'TestClass'

def test_case_0():
    var_0 = 'field1'
    var_1 = 1
    var_2 = None
    var_3 = 'BaseClass'
    var_4 = {}
    var_5 = 'SubClass'
    var_6 = '__weakref__'
    var_7 = '__weakref__'

def test_case_0():
    var_0 = 'field1'
    var_1 = 1
    var_2 = None
    var_3 = 'BaseClass'
    var_4 = 'field2'
    var_5 = 2
    var_6 = 'SubClass'
    var_7 = 'field1'
    var_8 = 'field2'



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_optional_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_nested_pclass. Retrieved 5/11 statements.
# Partially parsed test_serialize_preserves_field_order. Retrieved 6/12 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_complex_types. Retrieved 9/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 42
    var_4 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = 'json'
    var_3 = 'value'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 5
    var_4 = 10
    var_5 = 'inner'
    var_6 = 'b'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 42

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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_new_raises_invariant_exception_when_invariant_errors_exist. Retrieved 2/10 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_when_missing_fields_exist. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_when_both_invariant_errors_and_missing_fields. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'test_error'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'test_error'
    var_6 = 'TestClass.x'



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_pclassmeta_new_with_pclass_bases.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_set_method_iterates_over_pclass_fields. Retrieved 8/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = '_pclass_fields'
    var_8 = 'x'
    var_9 = 'y'
    var_10 = 'z'
    var_11 = bool(var_3)
    assert var_11 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_hash_basic. Retrieved 5/14 statements.
# Partially parsed test_hash_with_missing_fields. Retrieved 3/9 statements.
# Partially parsed test_hash_consistent. Retrieved 4/9 statements.
# Partially parsed test_hash_in_set. Retrieved 5/12 statements.
# Partially parsed test_hash_with_different_types. Retrieved 6/15 statements.
# Partially parsed test_hash_with_nested_structures. Retrieved 9/15 statements.
# Failed to parse test_hash_empty_pclass.
# Partially parsed test_hash_single_field. Retrieved 3/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'a'
    var_3 = 'b'
    var_4 = 1
    var_5 = 2

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'a'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = module_1.pmap(var_4)
    var_6 = 2
    var_7 = {var_2: var_3}
    var_8 = module_1.pmap(var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 43



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_basic. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_values. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_nested_pclass. Retrieved 5/11 statements.
# Partially parsed test_serialize_empty_pclass. Retrieved 1/5 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 2/9 statements.
# Partially parsed test_serialize_multiple_fields. Retrieved 8/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)

def test_case_0():
    var_0 = 42
    var_1 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 6/12 statements.
# Partially parsed test_pclass_missing_mandatory_field_raises_exception. Retrieved 1/6 statements.
# Partially parsed test_pclass_invariant_errors_or_missing_fields_true. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = 'test_error'
    var_3 = (var_1, var_2)
    var_4 = lambda obj: var_3
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_updates_data_and_marks_dirty. Retrieved 5/8 statements.
# Partially parsed test_set_with_same_value_does_not_mark_dirty. Retrieved 4/8 statements.
# Partially parsed test_set_replaces_existing_value. Retrieved 4/7 statements.
# Partially parsed test_set_returns_self_for_chaining. Retrieved 3/6 statements.
# Partially parsed test_set_with_none_value. Retrieved 3/6 statements.
# Partially parsed test_set_multiple_keys. Retrieved 7/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = 'key2'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'value1'
    var_5 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = 'value1'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = None
    var_4 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = 'key3'
    var_7 = 'value3'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 5/11 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value'
    var_3 = 10
    var_4 = {var_2: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = module_0.field()
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields_raises. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_multiple_fields. Retrieved 5/8 statements.
# Failed to parse test_pclass_new_empty_class.
# Partially parsed test_pclass_new_with_all_optional_fields. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -5
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_set_predicate_false_when_value_unchanged. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key1'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_cannot_set_after_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra_parameter. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_partial_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/11 statements.
# Failed to parse test_pclass_new_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

def test_case_0():
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 3
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclass_missing_mandatory_field_raises_invariant_exception. Retrieved 3/8 statements.
# Partially parsed test_pclass_field_invariant_error_raises_exception. Retrieved 2/7 statements.
# Partially parsed test_pclass_multiple_missing_mandatory_fields. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 42
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.mandatory_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = module_0.field()
    var_5 = 100
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestClass.field1'
    var_8 = 'TestClass.field2'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_set_predicate_false_when_value_unchanged. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key1'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_invariant_errors_exist. Retrieved 3/9 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_missing_mandatory_fields. Retrieved 1/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda val: (False, 'invariant_failed')
    var_2 = module_0.field(invariant=var_1)
    var_3 = 5
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_invariant_errors_present. Retrieved 9/18 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields_present. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'pyrsistent._pclass'
    var_2 = '_check_and_set_attr'
    var_3 = [var_2]
    var_4 = __import__(var_1, fromlist=var_3)
    var_5 = var_4._check_and_set_attr
    var_6 = False
    var_7 = 1
    var_8 = True
    assert var_8 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    assert var_3 is True
    var_4 = 'TestClass.x'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_meta_new_with_pclass_bases. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'TestPClass'
    var_3 = '__weakref__'
    var_4 = '_pclass_frozen'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 7/13 statements.
# Partially parsed test_pclass_constructor_without_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value'
    var_3 = 42
    var_4 = {var_2: var_3}
    var_5 = 'inner'
    var_6 = {var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 0
    var_2 = module_0.field(initial=var_1)
    var_3 = 1



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pclass_meta_new_with_pclass_bases. Retrieved 2/7 statements.


def test_case_0():
    var_0 = {}
    var_1 = ()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_hash_returns_consistent_hash_for_pclass_instances. Retrieved 4/12 statements.
# Partially parsed test_hash_differs_for_different_field_values. Retrieved 5/11 statements.
# Partially parsed test_hash_can_be_used_in_set. Retrieved 5/12 statements.
# Partially parsed test_hash_can_be_used_in_dict. Retrieved 3/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'value1'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/22 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 8/22 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_no_type_constraint. Retrieved 7/20 statements.
# Partially parsed test_check_and_set_attr_multiple_allowed_types. Retrieved 7/21 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_attr'
    var_6 = 42
    var_7 = bool(var_0 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = True
    var_3 = None
    var_4 = (var_2, var_3)
    var_5 = lambda x: var_4
    var_6 = 'test_attr'
    var_7 = 'invalid'
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = []
    var_1 = 'value_too_small'
    var_2 = False
    var_3 = (var_2, var_1)
    var_4 = lambda x: var_3
    var_5 = 'test_attr'
    var_6 = 42
    var_7 = bool(var_0 == [var_1])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True
    var_3 = (var_2, var_1)
    var_4 = lambda x: var_3
    var_5 = 'test_attr'
    var_6 = 'any_value'
    var_7 = bool(var_0 == [])
    assert var_7 is True

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'test_attr'
    var_6 = 'string_value'
    var_7 = bool(var_0 == [])
    assert var_7 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_reduce_returns_restore_pickle_and_class_data. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_repr_basic. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_string_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_mixed_types. Retrieved 6/10 statements.
# Partially parsed test_pclass_repr_with_initial_values. Retrieved 3/7 statements.
# Failed to parse test_pclass_repr_empty_class.
# Partially parsed test_pclass_repr_nested_values. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_with_none_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields_order. Retrieved 7/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'hello'
    var_3 = 'world'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 42
    var_4 = 'value'
    var_5 = True
    var_6 = 'MixedClass('
    var_7 = 'num=42'
    var_8 = "text='value'"
    var_9 = 'flag=True'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20
    var_4 = 'InitialClass('
    var_5 = 'x=10'
    var_6 = 'y=20'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'MultiField('
    var_7 = ')'
    var_8 = 'a=1'
    var_9 = 'b=2'
    var_10 = 'c=3'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_repr_format. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestClass('
    var_5 = 'x=1'
    var_6 = "y='hello'"
    var_7 = ')'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 7/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = '_pclass_fields'
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 'z'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_nested_objects. Retrieved 5/11 statements.
# Partially parsed test_serialize_preserves_field_order. Retrieved 6/12 statements.
# Partially parsed test_serialize_with_multiple_fields_some_missing. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 10
    var_4 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = None
    var_4 = module_0.field(initial=var_3)
    var_5 = 'value'
    var_6 = 'required'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_field_invariant_fails. Retrieved 2/7 statements.
# Partially parsed test_pclass_raises_invariant_exception_with_multiple_missing_fields. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.mandatory_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = module_0.field()
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestClass.mandatory_field1'
    var_8 = 'TestClass.mandatory_field2'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'error_code_123'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_eq_predicate_isinstance_true. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 3/16 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_value'
    var_2 = []
    var_3 = bool(var_2 == ['invariant_error_code'])
    assert var_3 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_pclass_hash_same_values_same_hash. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values_different_hash. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_hashable. Retrieved 2/7 statements.
# Partially parsed test_pclass_hash_can_be_used_in_set. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_can_be_used_as_dict_key. Retrieved 5/10 statements.
# Partially parsed test_pclass_hash_with_missing_values. Retrieved 4/12 statements.
# Partially parsed test_pclass_hash_multiple_fields. Retrieved 9/15 statements.
# Partially parsed test_pclass_hash_consistent. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'value1'
    var_4 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'test'
    var_5 = 2
    var_6 = 3
    var_7 = [var_3, var_5, var_6]
    var_8 = [var_3, var_5, var_6]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pclass_equality_predicate_at_line_3. Retrieved 4/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_wrong_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra_param. Retrieved 7/12 statements.
# Partially parsed test_pclass_new_with_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/9 statements.
# Failed to parse test_pclass_new_with_no_fields.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

def test_case_0():
    var_0 = 'string'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'Must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = True
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 2
    var_6 = {var_3: var_2, var_4: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 3
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = '10'
    var_4 = 20



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 8/11 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = {var_4}
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/14 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'invariant_error_code'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/9 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/10 statements.
# Partially parsed test_pclass_new_multiple_missing_fields. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_mixed_mandatory_optional. Retrieved 5/12 statements.
# Failed to parse test_pclass_new_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

def test_case_0():
    var_0 = 'string'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'y'
    var_5 = None



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_reduce_returns_restore_pickle_and_class_data. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/8 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = {var_4}
    var_6 = {var_1: var_2}
    var_7 = module_1.pmap(var_6)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '_pclass_frozen'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = var_1[0]
    assert var_5 == 'invariant_error'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_field_factory_and_ignore_extra. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_multiple_missing_mandatory_fields. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_preserves_field_order. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

def test_case_0():
    var_0 = '5'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

def test_case_0():
    var_0 = '10'
    var_1 = 'extra'
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 3
    var_4 = 1
    var_5 = 2



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_pclass_repr_with_string_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_nested_object. Retrieved 3/9 statements.
# Failed to parse test_pclass_repr_empty_class.
# Partially parsed test_pclass_repr_with_optional_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_repr_with_list_value. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_with_dict_value. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_with_none_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_boolean_values. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'MultiFieldClass('
    var_7 = 'x=1'
    var_8 = 'y=2'
    var_9 = 'z=3'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'OuterClass(inner=InnerClass(value=42))'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'DictClass(data='
    var_5 = "'key': 'value'"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = True
    var_3 = False
    var_4 = 'BoolClass('
    var_5 = 'flag1=True'
    var_6 = 'flag2=False'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_set_method_predicate_line_25. Retrieved 6/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10
    var_5 = '_pclass_fields'
    var_6 = bool(var_2)
    assert var_6 is True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_preserves_field_order. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_none_value. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 3/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'test_value'
    var_3 = bool(var_0 == ['invariant_error_code'])
    assert var_3 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_pclass_missing_fields_raises_exception. Retrieved 1/5 statements.
# Partially parsed test_pclass_both_invariant_errors_and_missing_fields. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda obj: (False, 'test_error')
    var_2 = (var_1,)
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = lambda obj: (False, 'invariant_failed')
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'invariant_failed'
    var_6 = 'TestClass.x'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_string_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_nested_structure. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_empty_pclass. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields_order. Retrieved 8/14 statements.
# Partially parsed test_pclass_repr_with_float_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_boolean_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_dict_value. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_class_name_in_output. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'TestClass('
    var_5 = 'x=1'
    var_6 = 'y=2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'TestClass('
    var_7 = ')'
    var_8 = 'a=1'
    var_9 = 'b=2'
    var_10 = 'c=3'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 3.14

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 'MyCustomClass'



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value'
    var_3 = 10
    var_4 = {var_2: var_3}
    var_5 = 'inner'
    var_6 = {var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_fields_with_initial. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 3



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_with_no_fields.
# Partially parsed test_pclass_constructor_multiple_instances_independent. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'TestClass.y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = None
    var_4 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/14 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 7/16 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 7/15 statements.
# Partially parsed test_check_and_set_attr_multiple_types. Retrieved 7/15 statements.
# Partially parsed test_check_and_set_attr_no_type_constraint. Retrieved 7/14 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'invalid'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = False
    var_1 = 'value_too_large'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == ['value_too_large'])
    assert var_7 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'hello'
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = (var_1, var_0)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'any_value'
    var_7 = bool(var_4 == [])
    assert var_7 is True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_fields_with_mixed_initial. Retrieved 5/8 statements.
# Failed to parse test_pclass_constructor_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 100
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 1
    var_6 = 3



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/14 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 7/15 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 7/15 statements.
# Partially parsed test_check_and_set_attr_multiple_types. Retrieved 7/15 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 7/14 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'not_an_int'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = False
    var_1 = 'value_too_small'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == ['value_too_small'])
    assert var_7 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'string_value'
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = (var_1, var_0)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'any_value'
    var_7 = bool(var_4 == [])
    assert var_7 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'data'
    var_2 = {var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_no_arguments. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_multiple_missing_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 20
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = {var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = module_0.field()
    var_5 = 30
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestClass.x'
    var_8 = 'TestClass.y'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = True
    var_3 = 2
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 1/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 999
    var_3 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_without_factory_fields_uses_raw_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

def test_case_0():
    var_0 = '42'
    var_1 = 'x'
    var_2 = {var_1}

def test_case_0():
    var_0 = 42
    var_1 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_all_fields_provided. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_pclass_frozen_attribute_set. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2
    var_4 = 'extra'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_cannot_set_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra_false. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_new_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_equality. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_empty_kwargs. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_complex_types. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'not among the specified fields'

def test_case_0():
    var_0 = 'invalid'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)

def test_case_0():
    var_0 = 1
    var_1 = 'hello'



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_eq_predicate_line_3_evaluates_to_true. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_partial_fields_with_defaults. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = 1



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 6/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x'
    var_7 = 'y'
    var_8 = 'z'



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/11 statements.
# Partially parsed test_pclass_hash_different_for_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_missing_values. Retrieved 3/8 statements.
# Partially parsed test_pclass_hash_is_hashable. Retrieved 2/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_repr_format. Retrieved 6/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestClass('
    var_5 = ')'
    var_6 = 'x=1'
    var_7 = "y='hello'"



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_field_invariant_fails. Retrieved 2/7 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_multiple_fields_missing. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'TestClass.mandatory_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: (x > 0, 'x must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.field1'
    var_6 = 'TestClass.field2'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_reduce_returns_restore_pickle_and_class_data. Retrieved 4/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_with_optional_field. Retrieved 6/14 statements.
# Partially parsed test_set_preserves_unmodified_fields. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10
    var_7 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 5
    var_4 = 'y'
    var_5 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 20



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_pclass_new_with_valid_field. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_multiple_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra_parameter. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 2/9 statements.
# Partially parsed test_pclass_new_preserves_field_order. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_default_initial_and_provided_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_none_value. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 99
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'are not among the specified fields'

def test_case_0():
    var_0 = 'string'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

def test_case_0():
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'x must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots_with_pclass_frozen. Retrieved 6/17 statements.


def test_case_0():
    var_0 = False
    var_1 = '_pclass_fields'
    var_2 = 'field1'
    var_3 = None
    var_4 = ()
    var_5 = 'TestClass'
    var_6 = '__slots__'
    var_7 = '_pclass_frozen'
    var_8 = 'field1'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_missing_fields. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 2
    var_4 = 'y'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 2/5 statements.
# Failed to parse test_pclass_constructor_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'y'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 5/21 statements.


def test_case_0():
    var_0 = []
    var_1 = 'value'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'invariant_error_code'
    var_5 = None



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 3/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'test_value'
    var_3 = bool(var_0 == ['invariant_error_code'])
    assert var_3 is True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_pclass_new_raises_invariant_exception_when_invariant_errors_exist. Retrieved 2/10 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_when_missing_fields. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_with_both_errors_and_missing. Retrieved 1/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_pclass_meta_new_basic. Retrieved 2/6 statements.
# Partially parsed test_pclass_meta_new_with_fields. Retrieved 3/12 statements.
# Partially parsed test_pclass_meta_new_slots_structure. Retrieved 4/9 statements.
# Partially parsed test_pclass_meta_new_weakref_only_on_base. Retrieved 4/10 statements.
# Partially parsed test_pclass_meta_new_invariant_storage. Retrieved 3/14 statements.
# Partially parsed test_pclass_meta_new_multiple_fields. Retrieved 4/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '_pclass_fields'
    var_3 = bool('_pclass_fields' in var_0)
    assert var_3 is True
    var_4 = '_pclass_invariants'
    var_5 = bool('_pclass_invariants' in var_0)
    assert var_5 is True
    var_6 = '__slots__'
    var_7 = bool('__slots__' in var_0)
    assert var_7 is True
    var_8 = '_pclass_frozen'
    var_9 = bool('_pclass_frozen' in var_0['__slots__'])
    assert var_9 is True

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'TestClassWithField'
    var_3 = '_pclass_fields'
    var_4 = 'test_field'
    var_5 = 'test_field'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClassSlots'
    var_2 = '__slots__'
    var_3 = var_0[var_2]
    var_4 = var_0['__slots__'][0]
    assert var_4 == '_pclass_frozen'
    var_5 = '__weakref__'
    var_6 = bool('__weakref__' in var_0['__slots__'])
    assert var_6 is True

def test_case_0():
    var_0 = {}
    var_1 = 'BaseClass'
    var_2 = '__weakref__'
    var_3 = bool('__weakref__' in var_0['__slots__'])
    assert var_3 is True
    var_4 = {}
    var_5 = 'DerivedClass'
    var_6 = '__weakref__'
    var_7 = bool('__weakref__' not in var_4['__slots__'])
    assert var_7 is True

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClassInvariant'
    var_2 = '_pclass_invariants'
    var_3 = '_pclass_invariants'

def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'TestClassMultipleFields'
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = '_pclass_frozen'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_set_method_predicate_line_25. Retrieved 9/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'x'
    var_7 = 10
    var_8 = '_pclass_fields'
    var_9 = 'x'
    var_10 = 'y'
    var_11 = 'z'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_pclass_eq_same_values. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 5/8 statements.
# Partially parsed test_pclass_eq_different_classes. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_with_non_pclass. Retrieved 4/6 statements.
# Partially parsed test_pclass_eq_missing_vs_present_field. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_reflexive. Retrieved 2/4 statements.
# Partially parsed test_pclass_eq_with_none_values. Retrieved 4/7 statements.
# Failed to parse test_pclass_eq_empty_classes.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2: var_1}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 1



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 1/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 5
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'

def test_case_0():
    var_0 = '5'



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_pclass_invariant_exception_raised_when_invariant_errors_exist. Retrieved 2/10 statements.
# Partially parsed test_pclass_invariant_exception_raised_when_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_invariant_exception_raised_with_both_errors_and_missing. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_multiple_types. Retrieved 6/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5
    var_4 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'test'
    var_4 = 42
    var_5 = True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 7/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = '_pclass_fields'
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 'z'



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_hash_same_values_same_hash. Retrieved 4/10 statements.
# Partially parsed test_hash_different_values_different_hash. Retrieved 5/11 statements.
# Partially parsed test_hash_with_missing_values. Retrieved 3/9 statements.
# Partially parsed test_hash_usable_in_set. Retrieved 3/10 statements.
# Partially parsed test_hash_usable_as_dict_key. Retrieved 3/8 statements.
# Partially parsed test_hash_with_multiple_fields. Retrieved 9/15 statements.
# Partially parsed test_hash_consistent_across_calls. Retrieved 2/7 statements.
# Partially parsed test_hash_with_none_values. Retrieved 3/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'value1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'test'
    var_5 = 2
    var_6 = 3
    var_7 = [var_3, var_5, var_6]
    var_8 = [var_3, var_5, var_6]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_frozen'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_repr_predicate. Retrieved 6/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestClass'
    var_5 = 'x=1'
    var_6 = "y='hello'"
    var_7 = 'TestClass('
    var_8 = ')'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raise_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_with_multiple_initial_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = {var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 3



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 5/18 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'test_value'
    var_4 = len(var_1)
    assert var_4 == 1
    var_5 = var_1[0]
    assert var_5 == 'error_code_1'



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 1/5 statements.
# Failed to parse test_pclass_constructor_empty_pclass.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

def test_case_0():
    var_0 = '5'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_true. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/17 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'test_value'
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'invariant_error_code'



# Parsed testcases at query #107
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_multiple_fields_with_mixed_setup. Retrieved 5/8 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_global_invariant_passes. Retrieved 4/10 statements.
# Failed to parse test_pclass_new_empty_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = True
    var_4 = module_0.field(mandatory=var_3)
    var_5 = 10
    var_6 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 2



# Parsed testcases at query #108
#--------------------------

# Partially parsed test_pclass_meta_new_creates_class_with_fields. Retrieved 4/15 statements.
# Partially parsed test_pclass_meta_new_sets_slots. Retrieved 3/13 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_for_direct_checkedtype_subclass. Retrieved 3/8 statements.
# Partially parsed test_pclass_meta_new_no_weakref_for_non_direct_checkedtype_subclass. Retrieved 3/7 statements.
# Partially parsed test_pclass_meta_new_stores_invariants. Retrieved 6/4 statements.
# Partially parsed test_pclass_meta_new_removes_field_from_dct. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'TestClass'
    var_3 = '_pclass_fields'
    var_4 = 'field1'
    var_5 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestClass'
    var_2 = '__slots__'
    var_3 = '_pclass_frozen'
    var_4 = 'field1'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__slots__'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__slots__'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = '_pclass_invariants'
    var_6 = bool(var_2)
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = '_pclass_invariants'
    var_6 = bool(var_2)
    assert var_6 is True

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestClass'
    var_2 = 'field1'
    var_3 = 'field1'



# Parsed testcases at query #109
#--------------------------

# Partially parsed test_set_method_iterates_over_pclass_fields. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 10



# Parsed testcases at query #110
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #111
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 5/10 statements.
# Partially parsed test_pclass_repr_string_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_empty_pclass. Retrieved 1/5 statements.
# Partially parsed test_pclass_repr_nested_structure. Retrieved 3/9 statements.
# Partially parsed test_pclass_repr_with_none_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_list_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_with_dict_field. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'TestClass('
    var_5 = 'x=1'
    var_6 = 'y=2'
    var_7 = ')'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'OuterClass('
    var_4 = 'InnerClass(value=42)'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'TestClass('



