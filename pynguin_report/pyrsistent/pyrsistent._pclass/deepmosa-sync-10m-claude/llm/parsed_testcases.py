####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pclass_meta_new_sets_pclass_fields. Retrieved 3/13 statements.
# Partially parsed test_pclass_meta_new_sets_pclass_invariants. Retrieved 5/4 statements.
# Partially parsed test_pclass_meta_new_sets_slots. Retrieved 2/11 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_slot_for_top_level. Retrieved 2/6 statements.
# Partially parsed test_pclass_meta_new_no_weakref_slot_for_subclass. Retrieved 2/5 statements.
# Partially parsed test_pclass_meta_new_removes_field_from_dct. Retrieved 2/11 statements.
# Partially parsed test_pclass_meta_new_returns_type_instance. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'TestClass'
    var_3 = '_pclass_fields'
    var_4 = 'field1'
    var_5 = 'field2'

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
    var_2 = '__slots__'
    var_3 = '_pclass_frozen'
    var_4 = 'field1'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = {}
    var_1 = 'TestSubClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestClass'
    var_2 = 'field1'
    var_3 = 'field1'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_new_key. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_different_value. Retrieved 4/7 statements.
# Partially parsed test_set_same_value_no_change. Retrieved 3/7 statements.
# Partially parsed test_set_multiple_keys. Retrieved 5/9 statements.
# Partially parsed test_set_returns_self_for_chaining. Retrieved 5/9 statements.
# Partially parsed test_set_with_none_value. Retrieved 3/6 statements.
# Partially parsed test_set_overwrites_previous_value. Retrieved 5/9 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'same_value'
    var_2 = 'key1'
    var_3 = {var_2: var_1}
    var_4 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = 'key1'
    var_7 = 'key2'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = 'value1'
    var_4 = 'key2'
    var_5 = 'value2'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key1'
    var_3 = None
    var_4 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'value2'
    var_5 = 'value3'
    var_6 = 'key1'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 5/8 statements.
# Partially parsed test_remove_nonexistent_item. Retrieved 4/8 statements.
# Partially parsed test_remove_item_that_was_set. Retrieved 5/9 statements.
# Partially parsed test_remove_multiple_items. Retrieved 7/11 statements.
# Partially parsed test_remove_via_delitem. Retrieved 3/7 statements.


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
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = 'key2'
    var_7 = 'key2'

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
    var_9 = 'key3'
    var_10 = 'key2'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key1'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_eq_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 5/9 statements.
# Partially parsed test_pclass_eq_different_types. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_non_pclass. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_missing_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_self. Retrieved 2/5 statements.
# Failed to parse test_pclass_eq_empty_classes.
# Partially parsed test_pclass_eq_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_pclass_eq_one_field_different. Retrieved 7/11 statements.


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
    var_1 = 1

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 999



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_violation. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_multiple_fields_with_mixed_initialization. Retrieved 4/8 statements.


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

def test_case_0():
    var_0 = 'invalid'
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
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = set()
    var_3 = True

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 123



# Parsed testcases at query #6
#--------------------------




def test_case_0():
    var_0 = '__weakref__'
    var_1 = '_pclass_frozen'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_persistent_returns_original_when_not_dirty. Retrieved 8/12 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 13/18 statements.
# Partially parsed test_persistent_passes_factory_fields_and_data. Retrieved 12/22 statements.
# Partially parsed test_persistent_with_removed_field. Retrieved 13/18 statements.
# Partially parsed test_persistent_multiple_calls_after_set. Retrieved 10/16 statements.


def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = '__init__'
    var_3 = 'data'
    var_4 = '_factory_fields'
    var_5 = lambda self, _factory_fields=None, **kwargs: setattr(self, var_3, kwargs) or setattr(self, var_4, _factory_fields)
    var_6 = {var_2: var_5}
    var_7 = [var_0, var_1, var_6]
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 1
    var_11 = 2
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = 10

def test_case_0():
    var_0 = {}
    var_1 = 'MockPClass'
    var_2 = ()
    var_3 = '__init__'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 1
    var_7 = 2
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 10
    var_10 = 'c'
    var_11 = 3
    var_12 = var_0['_factory_fields']
    var_13 = bool(var_0['_factory_fields'] == {'a', 'c'})
    assert var_13 is True
    var_14 = var_0['kwargs']
    var_15 = bool(var_0['kwargs'] == {'a': 10, 'b': 2, 'c': 3})
    assert var_15 is True

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = '__init__'
    var_3 = None
    var_4 = lambda self, _factory_fields=None, **kwargs: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'c'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = {var_7: var_10, var_8: var_11, var_9: var_12}
    var_14 = 'b'

def test_case_0():
    var_0 = 'MockPClass'
    var_1 = ()
    var_2 = '__init__'
    var_3 = None
    var_4 = lambda self, _factory_fields=None, **kwargs: var_3
    var_5 = {var_2: var_4}
    var_6 = [var_0, var_1, var_5]
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = 5



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_preserves_original. Retrieved 5/9 statements.
# Partially parsed test_set_with_optional_fields. Retrieved 4/8 statements.
# Partially parsed test_set_mixed_args_and_kwargs. Retrieved 7/11 statements.


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
    var_3 = 2
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 0
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 10
    var_6 = 20



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_meta_weakref_slot_added_when_bases_is_pclass. Retrieved 2/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'TestPClass'
    var_2 = '__weakref__'
    var_3 = '_pclass_frozen'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_with_only_defined_fields. Retrieved 6/10 statements.
# Partially parsed test_pclass_reduce_empty_instance. Retrieved 2/7 statements.


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
    var_2 = None
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_invariant_errors_present. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields_present. Retrieved 3/8 statements.
# Partially parsed test_pclass_predicate_at_line_25_true_with_missing_mandatory_field. Retrieved 5/9 statements.


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
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'TestClass.y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = False
    var_4 = 'value'
    var_5 = True
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_is_pclass_returns_false_for_empty_bases. Retrieved 7/9 statements.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = {}
    var_3 = ()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = '__weakref__'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_preserves_all_fields. Retrieved 7/11 statements.
# Partially parsed test_set_with_single_arg_and_value. Retrieved 4/8 statements.
# Partially parsed test_set_with_optional_fields. Retrieved 4/8 statements.


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
    var_6 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'original'
    var_2 = 'name'
    var_3 = 'updated'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 10



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_hash_same_values. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_missing_values. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_hashable. Retrieved 4/9 statements.
# Partially parsed test_pclass_hash_in_set. Retrieved 5/12 statements.
# Partially parsed test_pclass_hash_in_dict. Retrieved 5/12 statements.
# Partially parsed test_pclass_hash_single_field. Retrieved 2/8 statements.
# Partially parsed test_pclass_hash_with_string_values. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_with_none_values. Retrieved 2/8 statements.


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
    var_4 = 'value1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 'data'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)
    var_2 = None
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_initial_value. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_nested_pclass. Retrieved 5/11 statements.
# Partially parsed test_serialize_returns_dict. Retrieved 2/7 statements.
# Partially parsed test_serialize_does_not_modify_original. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_boolean_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

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
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 5
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = True
    var_3 = False



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_default_value. Retrieved 3/6 statements.


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
    var_0 = '42'

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
    var_0 = 100
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_optional_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 5/9 statements.
# Partially parsed test_serialize_returns_dict. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_multiple_fields. Retrieved 8/12 statements.


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
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 100
    var_4 = 'json'

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
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 'd'



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_pclass_meta_new_with_pclass_bases.




# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_invariant_errors_exist. Retrieved 4/15 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields_exist. Retrieved 3/7 statements.
# Partially parsed test_pclass_raises_invariant_exception_with_both_invariant_errors_and_missing_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = 1
    var_3 = True
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = False
    var_5 = True
    assert var_5 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_pclass_repr_string_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_nested_pclass. Retrieved 3/9 statements.
# Partially parsed test_pclass_repr_with_none. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_list. Retrieved 5/9 statements.
# Failed to parse test_pclass_repr_empty_pclass.
# Partially parsed test_pclass_repr_optional_field_not_set. Retrieved 3/7 statements.


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
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 'OptionalClass('
    var_5 = 'x=1'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_meta_weakref_not_added_when_not_pclass. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = {}
    var_4 = ()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '__weakref__'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_repr_format. Retrieved 6/12 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_repr_format. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestPClass('
    var_5 = 'x=1'
    var_6 = "y='hello'"
    var_7 = ')'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_becomes_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
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
    var_3 = 20

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
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

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
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 3/7 statements.


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
    var_3 = 'y'

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 7/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_fields'
    var_5 = []
    var_6 = 'x'
    var_7 = bool('x' in var_5)
    assert var_7 is True
    var_8 = 'y'
    var_9 = bool('y' in var_5)
    assert var_9 is True
    var_10 = len(var_5)
    assert var_10 == 2



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = '_pclass_frozen'



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_mandatory_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_field_invariant_fails. Retrieved 3/8 statements.
# Partially parsed test_pclass_predicate_line_25_true_with_missing_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_predicate_line_25_true_with_invariant_errors. Retrieved 9/16 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = set()
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    var_4 = bool(var_3)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = 1
    var_3 = 'test_error'
    var_4 = [var_3]
    var_5 = tuple(var_4)
    var_6 = ()
    var_7 = 'Field invariant failed'
    var_8 = [var_7]
    var_9 = True
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_repr_format. Retrieved 6/12 statements.


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



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
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
    var_5 = 'TestClass'

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
    var_2 = True
    var_3 = 2
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



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pclass_eq_same_class_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_same_class_different_values. Retrieved 5/9 statements.
# Partially parsed test_pclass_eq_same_class_one_field_missing. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_classes. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_non_pclass_object. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_multiple_fields. Retrieved 9/13 statements.
# Partially parsed test_pclass_eq_reflexive. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_symmetric. Retrieved 2/6 statements.


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
    var_4 = None

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
    var_1 = 1



# Parsed testcases at query #35
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



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_new_empty.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/10 statements.
# Partially parsed test_pclass_new_with_failing_global_invariant. Retrieved 4/12 statements.


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
    var_2 = 3
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'
    var_5 = 'not among the specified fields'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '42'

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
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -5
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_partial_initialization. Retrieved 3/6 statements.


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
    var_2 = True
    var_3 = 2
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
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 10



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_hash_returns_consistent_hash_for_same_pclass. Retrieved 4/11 statements.
# Partially parsed test_hash_differs_for_different_pclass_values. Retrieved 5/11 statements.
# Partially parsed test_hash_pclass_with_single_field. Retrieved 2/7 statements.
# Partially parsed test_hash_pclass_with_optional_fields. Retrieved 4/10 statements.
# Partially parsed test_hash_pclass_hashable_in_set. Retrieved 3/12 statements.
# Partially parsed test_hash_pclass_hashable_in_dict. Retrieved 5/10 statements.


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
    var_4 = None

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



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_reduce_returns_restore_pickle_and_class_data. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 8/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 'y'
    var_6 = 0
    var_7 = 1
    var_8 = var_6 + var_7
    assert var_8 == 2
    var_9 = 'factory'



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pclass_new_basic_field_assignment. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_cannot_modify_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/11 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/14 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_partial_field_initialization. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_none_value. Retrieved 2/5 statements.
# Failed to parse test_pclass_new_empty_pclass.


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
    var_0 = 'not_an_int'
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
    var_5 = 'x_must_be_greater_than_y'

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
    var_1 = 20
    var_2 = module_0.field(initial=var_1)
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '42'
    var_2 = 100
    var_3 = 'x'
    var_4 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_default_factory. Retrieved 1/5 statements.
# Failed to parse test_pclass_constructor_empty_class.


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
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'
    var_6 = 'TestClass.y'

def test_case_0():
    var_0 = '42'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_set_method_iterates_over_pclass_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_reduce_returns_tuple_with_restore_pickle_and_class_data. Retrieved 4/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_field_invariant_fails. Retrieved 6/13 statements.
# Partially parsed test_pclass_raises_invariant_exception_with_both_errors_and_missing. Retrieved 2/7 statements.


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
    var_1 = module_0.field()
    var_2 = False
    var_3 = 'invariant_error'
    var_4 = (var_2, var_3)
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True

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



# Parsed testcases at query #46
#--------------------------

# Failed to parse test_pclass_meta_new_with_pclass_bases.




# Parsed testcases at query #47
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_missing_fields. Retrieved 2/6 statements.
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
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

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
    var_1 = 5



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pclass_meta_new_with_single_checkedtype_base. Retrieved 11/26 statements.
# Partially parsed test_pclass_meta_new_with_multiple_bases. Retrieved 3/17 statements.
# Partially parsed test_pclass_meta_new_fields_removed_from_dct. Retrieved 3/15 statements.
# Partially parsed test_pclass_meta_new_invariants_stored. Retrieved 6/4 statements.
# Partially parsed test_pclass_meta_new_slots_includes_fields_and_frozen. Retrieved 3/15 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '__invariant__'
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda self: var_5
    var_7 = 'TestPClass'
    var_8 = '_pclass_fields'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = '_pclass_invariants'
    var_12 = '__slots__'
    var_13 = '_pclass_frozen'
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = '__weakref__'

def test_case_0():
    var_0 = 'child_field'
    var_1 = 'ChildPClass'
    var_2 = '_pclass_fields'
    var_3 = 'child_field'
    var_4 = '__weakref__'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestPClass'
    var_2 = 'field1'
    var_3 = '_pclass_fields'
    var_4 = 'field1'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestPClass'
    var_5 = '_pclass_invariants'
    var_6 = bool(var_2)
    assert var_6 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestPClass'
    var_5 = '_pclass_invariants'
    var_6 = bool(var_2)
    assert var_6 is True

def test_case_0():
    var_0 = 'field_a'
    var_1 = 'field_b'
    var_2 = 'TestPClass'
    var_3 = '_pclass_frozen'
    var_4 = 'field_a'
    var_5 = 'field_b'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_hash_basic. Retrieved 4/10 statements.
# Partially parsed test_hash_different_values. Retrieved 5/11 statements.
# Partially parsed test_hash_single_field. Retrieved 2/8 statements.
# Partially parsed test_hash_with_optional_fields. Retrieved 3/9 statements.
# Partially parsed test_hash_in_set. Retrieved 3/10 statements.
# Partially parsed test_hash_in_dict. Retrieved 3/10 statements.
# Partially parsed test_hash_with_multiple_fields_different_order. Retrieved 6/12 statements.
# Partially parsed test_hash_consistency. Retrieved 2/7 statements.
# Partially parsed test_hash_with_string_values. Retrieved 4/10 statements.
# Partially parsed test_hash_with_none_values. Retrieved 2/8 statements.


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
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 'data'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_single_field. Retrieved 2/7 statements.


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
    var_4 = 'x'
    var_5 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Failed to parse test_pclass_constructor_with_no_fields.
# Partially parsed test_pclass_constructor_with_multiple_mandatory_fields_missing. Retrieved 2/6 statements.


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
    var_1 = 5

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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 6/9 statements.
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
    var_1 = 'x'
    var_2 = [var_1]
    var_3 = set(var_2)
    var_4 = 5

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_hash_returns_consistent_value. Retrieved 4/13 statements.
# Partially parsed test_hash_different_for_different_values. Retrieved 3/9 statements.
# Partially parsed test_hash_with_multiple_fields. Retrieved 6/12 statements.
# Partially parsed test_hash_with_missing_values. Retrieved 2/8 statements.


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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 10
    var_4 = 20
    var_5 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_without_factory_fields. Retrieved 2/6 statements.
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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 2/10 statements.
# Partially parsed test_pclass_missing_mandatory_field_raises_exception. Retrieved 1/6 statements.
# Partially parsed test_pclass_both_invariant_and_missing_field_errors. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Test invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestPClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invariant error'
    var_6 = 'TestPClass.x'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_multiple_field_invariants_fail. Retrieved 2/11 statements.
# Partially parsed test_pclass_new_empty_instance. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_multiple_valid_types. Retrieved 2/7 statements.


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

def test_case_0():
    var_0 = '5'

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
    var_2 = 2
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '5'
    var_2 = 10
    var_3 = 'x'
    var_4 = {var_3}

def test_case_0():
    var_0 = -1
    var_1 = -2
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 5
    var_1 = 'hello'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty.
# Partially parsed test_pclass_constructor_multiple_fields_with_initial. Retrieved 4/7 statements.


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
    var_1 = 5
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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_multiple_mandatory_fields_missing. Retrieved 2/6 statements.


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
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'TestClass.x'

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



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 6/22 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/19 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 3/17 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 3/16 statements.
# Partially parsed test_check_and_set_attr_multiple_valid_types. Retrieved 3/16 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = 'test_field'
    var_5 = 42
    var_6 = bool(var_3 == [])
    assert var_6 is True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'invalid'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 42
    var_3 = bool(var_0 == ['invariant_error_code'])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'any_value'
    var_3 = bool(var_0 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'string_value'
    var_3 = bool(var_0 == [])
    assert var_3 is True



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/17 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'test_value'
    var_2 = []
    var_3 = len(var_2)
    assert var_3 == 1
    var_4 = var_2[0]
    assert var_4 == 'invariant_error_code'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
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
    var_4 = 'not among the specified fields'

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
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_name'
    var_2 = 42
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'invariant_error_code'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 6/22 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error_code_1'
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = 'test_field'
    var_5 = 'test_value'
    var_6 = bool(var_3 == ['error_code_1'])
    assert var_6 is True



# Parsed testcases at query #65
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



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 9/24 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 8/21 statements.
# Partially parsed test_check_and_set_attr_multiple_types. Retrieved 8/20 statements.
# Partially parsed test_check_and_set_attr_no_type_constraint. Retrieved 8/19 statements.


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
    var_9 = 'test_field'
    var_10 = bool(var_1 == [])
    assert var_10 is True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = False
    var_3 = 'value must be positive'
    var_4 = (var_2, var_3)
    var_5 = lambda x: var_4
    var_6 = 'test_field'
    var_7 = 42
    var_8 = bool(var_1 == ['value must be positive'])
    assert var_8 is True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = True
    var_3 = None
    var_4 = (var_2, var_3)
    var_5 = lambda x: var_4
    var_6 = 'test_field'
    var_7 = 'hello'
    var_8 = bool(var_1 == [])
    assert var_8 is True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = []
    var_2 = None
    var_3 = True
    var_4 = (var_3, var_2)
    var_5 = lambda x: var_4
    var_6 = 'test_field'
    var_7 = 'any_value'
    var_8 = bool(var_1 == [])
    assert var_8 is True



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/6 statements.
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
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

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
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 15



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
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
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_override_initial_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_preserves_type. Retrieved 2/6 statements.


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
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)
    var_4 = 3
    var_5 = module_0.field(initial=var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42



# Parsed testcases at query #70
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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 2/5 statements.
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
    var_1 = set()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'y'

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
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_rejected. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
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



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 4/17 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/18 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/17 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 4/16 statements.
# Partially parsed test_check_and_set_attr_multiple_allowed_types. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42
    var_4 = bool(var_1 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42
    var_4 = bool(var_1 == ['Value must be positive'])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'any_value'
    var_4 = bool(var_1 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'string_value'
    var_4 = bool(var_1 == [])
    assert var_4 is True



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_partial_fields. Retrieved 3/6 statements.


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
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}

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
    var_1 = 0
    var_2 = module_0.field(initial=var_1)
    var_3 = 5



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
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
    var_3 = 20

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
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '5'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 2/6 statements.
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
    var_1 = 5
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



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots. Retrieved 4/11 statements.
# Partially parsed test_pclass_meta_new_moves_pfield_to_pclass_fields. Retrieved 3/11 statements.
# Partially parsed test_pclass_meta_new_with_weakref_slot. Retrieved 2/6 statements.
# Partially parsed test_pclass_meta_new_without_weakref_slot. Retrieved 2/5 statements.
# Partially parsed test_pclass_meta_new_stores_invariants. Retrieved 6/4 statements.
# Partially parsed test_pclass_meta_new_multiple_fields. Retrieved 4/11 statements.


def test_case_0():
    var_0 = None
    var_1 = 'test_attr'
    var_2 = 'TestClass'
    var_3 = '__slots__'
    var_4 = '_pclass_frozen'

def test_case_0():
    var_0 = None
    var_1 = 'my_field'
    var_2 = 'TestClass'
    var_3 = 'my_field'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = '_pclass_invariants'

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = '_pclass_invariants'

def test_case_0():
    var_0 = None
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'TestClass'
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'field1'
    var_7 = 'field2'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_with_missing_value. Retrieved 4/8 statements.
# Partially parsed test_set_preserves_all_fields. Retrieved 7/11 statements.


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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_freezes_instance. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_cannot_set_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_multiple_fields_and_initial. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_without_factory_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_ignore_extra_true. Retrieved 2/5 statements.


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
    var_4 = 'are not among the specified fields'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = -1
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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = module_0.field(initial=var_1)
    var_3 = 200
    var_4 = module_0.field(initial=var_3)
    var_5 = 1

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
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = '10'
    var_4 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_with_no_fields. Retrieved 3/9 statements.
# Partially parsed test_pclass_reduce_preserves_all_attributes. Retrieved 9/13 statements.


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
    var_1 = 100
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'hello'
    var_4 = 42
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_args. Retrieved 6/10 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_preserves_unmodified_fields. Retrieved 7/11 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_with_field_factory. Retrieved 3/7 statements.
# Partially parsed test_set_original_unchanged. Retrieved 6/12 statements.


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
    var_6 = 10

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
    var_2 = '2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 100
    var_5 = 200



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.x'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_empty. Retrieved 1/5 statements.
# Partially parsed test_pclass_repr_nested. Retrieved 3/9 statements.
# Partially parsed test_pclass_repr_with_special_characters. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_numeric_types. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_with_boolean. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_list. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)
    var_2 = 'EmptyClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'Outer'
    var_4 = 'Inner'
    var_5 = 'value=42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello\'world"test'
    var_2 = 'SpecialClass'
    var_3 = 'text='

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 3.14
    var_4 = 'integer=42'
    var_5 = 'floating=3.14'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'flag=True'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'ListClass'
    var_6 = 'items='
    var_7 = '[1, 2, 3]'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
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
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool('mandatory' in str(e).lower() or 'invariant' in str(e).lower())
    assert var_4 is True

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
    var_3 = 'frozen'
    var_4 = bool('frozen' in str(e).lower())
    assert var_4 is True

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
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_meta_new_weakref_slot_added_when_is_pclass_true. Retrieved 5/14 statements.
# Partially parsed test_pclass_meta_new_weakref_slot_not_added_when_is_pclass_false. Retrieved 5/13 statements.


def test_case_0():
    var_0 = True
    var_1 = '_pclass_fields'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'TestPClass'
    var_5 = '__weakref__'
    var_6 = '_pclass_frozen'

def test_case_0():
    var_0 = False
    var_1 = '_pclass_fields'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = 'TestPClass'
    var_5 = '__weakref__'
    var_6 = '_pclass_frozen'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 3/20 statements.
# Partially parsed test_pclass_missing_mandatory_field_raises_exception. Retrieved 1/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots_for_fields. Retrieved 5/13 statements.
# Partially parsed test_pclass_meta_new_adds_weakref_for_direct_checkedtype_subclass. Retrieved 2/6 statements.
# Partially parsed test_pclass_meta_new_no_weakref_for_indirect_subclass. Retrieved 6/15 statements.
# Partially parsed test_pclass_meta_new_stores_invariants. Retrieved 2/11 statements.
# Partially parsed test_pclass_meta_new_moves_fields_to_pclass_fields. Retrieved 4/13 statements.
# Partially parsed test_pclass_meta_new_inherits_parent_fields. Retrieved 6/17 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'TestClass'
    var_5 = '_pclass_frozen'
    var_6 = 'field1'
    var_7 = 'field2'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = 'ParentClass'
    var_4 = {}
    var_5 = 'ChildClass'
    var_6 = '__weakref__'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = '_pclass_invariants'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = 'TestClass'
    var_4 = 'field1'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = 'ParentClass'
    var_4 = 'field2'
    var_5 = 'ChildClass'
    var_6 = 'field1'
    var_7 = 'field2'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_repr_format. Retrieved 6/12 statements.


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



# Parsed testcases at query #14
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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_meta_weakref_not_added_when_not_pclass_bases. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = '_pclass_fields'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '__weakref__'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 3/10 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/13 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 3/16 statements.
# Partially parsed test_check_and_set_attr_passed_invariant. Retrieved 3/14 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = []
    var_2 = 'test_value'
    var_3 = bool(var_1 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'name'
    var_1 = []
    var_2 = 'name'
    var_3 = 123
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'count'
    var_1 = []
    var_2 = -5
    var_3 = bool(var_1 == ['must_be_positive'])
    assert var_3 is True

def test_case_0():
    var_0 = 'count'
    var_1 = []
    var_2 = 5
    var_3 = bool(var_1 == [])
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value'
    var_2 = []
    var_3 = 'any_value'
    var_4 = bool(var_2 == [])
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_invariant_errors_present. Retrieved 6/12 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 1/6 statements.
# Partially parsed test_pclass_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 5/11 statements.


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
    var_7 = 'test_error'

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
    var_2 = False
    var_3 = 'invariant_error'
    var_4 = (var_2, var_3)
    var_5 = lambda obj: var_4
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'invariant_error'
    var_8 = 'TestClass.x'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_field_invariant_violation. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra_parameter. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_multiple_fields. Retrieved 7/10 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_inherits_frozen_attribute. Retrieved 4/8 statements.
# Failed to parse test_pclass_new_with_no_fields.
# Partially parsed test_pclass_new_with_multiple_invariant_errors. Retrieved 2/11 statements.


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

def test_case_0():
    var_0 = '42'

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '10'
    var_2 = 20
    var_3 = 'x'
    var_4 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 30
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 1
    var_6 = 2
    var_7 = 4

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = '_pclass_frozen'
    var_3 = False

def test_case_0():
    var_0 = -1
    var_1 = -2
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_hash_same_values_same_hash. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values_different_hash. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_usable_in_set. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_usable_as_dict_key. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_with_string_fields. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_with_nested_values. Retrieved 10/16 statements.


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
    var_2 = 'test'
    var_3 = 'desc'

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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_only_assigned_fields. Retrieved 6/10 statements.


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
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_field_invariant_violation. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_multiple_type_options. Retrieved 2/7 statements.
# Failed to parse test_pclass_new_empty_class.
# Partially parsed test_pclass_new_with_ignore_extra_parameter. Retrieved 4/7 statements.


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
    var_4 = "'z' are not among the specified fields"

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
    var_2 = 3
    var_3 = 4
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum must be 10'

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
    var_0 = 1
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = set()
    var_3 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 5/8 statements.
# Partially parsed test_remove_nonexistent_item. Retrieved 4/8 statements.
# Partially parsed test_remove_after_set. Retrieved 5/9 statements.
# Partially parsed test_remove_discards_from_factory_fields. Retrieved 4/8 statements.
# Partially parsed test_remove_multiple_items. Retrieved 7/11 statements.


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
    var_4 = 'nonexistent_key'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = 'key2'
    var_7 = 'key2'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = 'key1'
    var_6 = 'key1'
    var_7 = 'key1'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'key2'
    var_3 = 'key3'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'value3'
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_pclass_repr_with_string_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_none_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_list_value. Retrieved 5/9 statements.
# Failed to parse test_pclass_repr_empty_pclass.
# Partially parsed test_pclass_repr_with_initial_value_not_set. Retrieved 3/7 statements.


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
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 5
    var_4 = 'OptionalFieldClass('
    var_5 = 'required=5'
    var_6 = 'optional=None'



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_optional_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_nested_pclass. Retrieved 3/9 statements.
# Partially parsed test_serialize_with_multiple_fields_and_types. Retrieved 13/17 statements.


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

def test_case_0():
    var_0 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 42
    var_3 = 'inner'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 123
    var_5 = 'test'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'key'
    var_11 = 'value'
    var_12 = {var_10: var_11}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_hash_same_values_same_hash. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_values_different_hash. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_hashable_in_set. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_hashable_as_dict_key. Retrieved 5/10 statements.
# Partially parsed test_pclass_hash_with_missing_values. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_consistent. Retrieved 4/9 statements.
# Partially parsed test_pclass_hash_with_nested_values. Retrieved 7/13 statements.
# Partially parsed test_pclass_hash_with_string_values. Retrieved 2/8 statements.


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
    var_5 = (var_2, var_3, var_4)
    var_6 = (var_2, var_3, var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pclass_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_multiple_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_with_string_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_nested_object. Retrieved 3/9 statements.
# Partially parsed test_pclass_repr_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_repr_with_list_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_repr_with_dict_field. Retrieved 4/8 statements.
# Failed to parse test_pclass_repr_empty_class.
# Partially parsed test_pclass_repr_with_boolean_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_none_field. Retrieved 2/6 statements.


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
    var_1 = None
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 'x=1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'ListClass(items=[1, 2, 3])'

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
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/8 statements.
# Partially parsed test_set_preserves_other_fields. Retrieved 7/11 statements.
# Partially parsed test_set_with_string_key_and_value. Retrieved 4/8 statements.
# Partially parsed test_set_with_complex_values. Retrieved 9/13 statements.


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
    var_6 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'old'
    var_2 = 'name'
    var_3 = 'new'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/22 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/19 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/18 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 4/16 statements.
# Partially parsed test_check_and_set_attr_multiple_valid_types. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42
    var_4 = bool(var_1 == ['value_too_small'])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'any_value'
    var_4 = bool(var_1 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'string_value'
    var_4 = bool(var_1 == [])
    assert var_4 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 3/7 statements.
# Partially parsed test_pclass_invariant_errors_with_invariant_function. Retrieved 2/11 statements.
# Partially parsed test_pclass_missing_mandatory_field. Retrieved 3/7 statements.


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
    var_2 = module_0.field()
    var_3 = 25
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestClass.name'



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_serialize_with_no_fields.
# Partially parsed test_serialize_with_simple_fields. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_optional_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_nested_pclass. Retrieved 5/11 statements.
# Partially parsed test_serialize_excludes_missing_values. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_serialize_preserves_field_values. Retrieved 6/10 statements.


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
    var_3 = 5

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
    var_5 = 'outer_value'
    var_6 = 'nested'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.field(initial=var_0)
    var_2 = None
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 30
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'test'
    var_4 = 42
    var_5 = True



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_meta_new_basic. Retrieved 3/13 statements.
# Partially parsed test_pclass_meta_new_with_fields. Retrieved 2/15 statements.
# Failed to parse test_pclass_meta_new_slots_includes_fields.
# Failed to parse test_pclass_meta_new_inherited_fields.
# Partially parsed test_pclass_meta_new_invariants. Retrieved 4/4 statements.
# Failed to parse test_pclass_meta_new_no_weakref_for_non_direct_subclass.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'
    var_3 = '_pclass_frozen'
    var_4 = '__weakref__'

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'x'
    var_3 = 'y'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = '_pclass_invariants'
    var_4 = bool(var_1)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = '_pclass_invariants'
    var_4 = bool(var_1)
    assert var_4 is True

def test_case_0():
    var_0 = '_pclass_frozen'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pclass_meta_new_adds_weakref_slot_when_is_pclass_true. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = '_pclass_fields'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '__weakref__'
    var_5 = '_pclass_frozen'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 1/5 statements.
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

def test_case_0():
    var_0 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = True
    var_4 = module_0.field(mandatory=var_3)
    var_5 = 1
    var_6 = 3



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_hash. Retrieved 4/13 statements.
# Partially parsed test_pclass_hash_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_optional_fields. Retrieved 4/12 statements.
# Partially parsed test_pclass_hash_usable_in_set. Retrieved 5/17 statements.
# Partially parsed test_pclass_hash_usable_in_dict. Retrieved 5/11 statements.


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
    var_4 = None

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
    var_2 = 2
    var_3 = 'value1'
    var_4 = 'value2'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pclass_invariant_errors_raises_exception. Retrieved 1/6 statements.
# Partially parsed test_pclass_missing_fields_raises_exception. Retrieved 3/8 statements.
# Partially parsed test_pclass_invariant_errors_or_missing_fields_true. Retrieved 3/7 statements.


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
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = 1
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'TestClass.y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = False
    var_3 = True
    assert var_3 is True



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_remove_item_exists_in_data. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'key1'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pclass_meta_weakref_not_added_when_not_pclass_bases. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = {}
    var_4 = ()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '__weakref__'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty_object. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_single_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_reduce_with_complex_values. Retrieved 11/16 statements.


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
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'key'
    var_8 = 'value'
    var_9 = {var_7: var_8}
    var_10 = (var_3, var_4)



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Failed to parse test_pclass_reduce_with_no_fields.
# Partially parsed test_pclass_reduce_with_multiple_types. Retrieved 9/14 statements.


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
    var_3 = 'string'
    var_4 = 123
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 7/11 statements.
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
    var_4 = 'y'

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_pclass_eq_same_class_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_same_class_different_values. Retrieved 5/9 statements.
# Partially parsed test_pclass_eq_different_class. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_missing_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_with_non_pclass_object. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_reflexive. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_with_none_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_complex_values. Retrieved 10/14 statements.


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
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1
    var_4 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2: var_1}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = None

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



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_default_and_provided_values. Retrieved 3/6 statements.


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
    var_4 = 'y'

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
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

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
    var_3 = module_0.field(initial=var_2)
    var_4 = 100



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pclass_new_iterates_over_pclass_fields. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = '_pclass_frozen'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 5/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = '_pclass_fields'
    var_5 = 'x'
    var_6 = 'y'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_hash_basic. Retrieved 5/14 statements.
# Partially parsed test_hash_with_missing_fields. Retrieved 4/10 statements.
# Partially parsed test_hash_consistent. Retrieved 4/9 statements.
# Partially parsed test_hash_different_field_values. Retrieved 4/10 statements.
# Partially parsed test_hash_with_string_fields. Retrieved 4/10 statements.
# Partially parsed test_hash_hashable_in_set. Retrieved 3/10 statements.
# Partially parsed test_hash_hashable_in_dict. Retrieved 3/10 statements.


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
    var_4 = None

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
    var_2 = 'test'
    var_3 = 'data'

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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_pclass_eq_same_class_equal_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_same_class_different_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_eq_same_class_one_field_missing. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_different_classes. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_with_none_values. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_with_complex_values. Retrieved 10/13 statements.
# Partially parsed test_pclass_eq_returns_not_implemented_for_different_type. Retrieved 3/6 statements.
# Partially parsed test_pclass_eq_single_field. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_multiple_fields_all_match. Retrieved 8/11 statements.
# Partially parsed test_pclass_eq_multiple_fields_one_differs. Retrieved 9/12 statements.


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
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 2

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'not a pclass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4
    var_8 = 5



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pclass_meta_new_predicate_line_1. Retrieved 1/8 statements.


def test_case_0():
    var_0 = '__new__'



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.
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

def test_case_0():
    var_0 = '5'

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



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_mandatory_field. Retrieved 1/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_pclass_new_raises_invariant_exception_when_invariant_errors_exist. Retrieved 2/10 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_when_missing_mandatory_field. Retrieved 1/6 statements.
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
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #52
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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_serialize_iterates_over_pclass_fields. Retrieved 11/20 statements.


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
    var_10 = [var_7, var_8, var_9]



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 6/20 statements.


def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'error_code_1'
    var_3 = (var_1, var_2)
    var_4 = 'test_field'
    var_5 = 'test_value'
    var_6 = bool(var_0 == ['error_code_1'])
    assert var_6 is True



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
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
    var_1 = 5
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
    var_1 = module_0.field()
    var_2 = None
    var_3 = 5



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 4/7 statements.


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
    var_4 = 'extra_field'

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
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 1/7 statements.
# Partially parsed test_pclass_constructor_multiple_mandatory_fields_missing. Retrieved 4/9 statements.


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

def test_case_0():
    var_0 = '5'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = True
    var_3 = module_0.field(mandatory=var_2)
    var_4 = module_0.field()
    var_5 = 3
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestClass.x'
    var_8 = 'TestClass.y'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_remove_item_exists. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'key1'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/22 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/19 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 4/18 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 4/16 statements.
# Partially parsed test_check_and_set_attr_multiple_valid_types. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42
    var_4 = bool(var_1 == ['invariant_error_code'])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'any_value'
    var_4 = bool(var_1 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 3.14
    var_4 = bool(var_1 == [])
    assert var_4 is True



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 5/8 statements.
# Partially parsed test_remove_nonexistent_item. Retrieved 4/8 statements.
# Partially parsed test_remove_item_after_set. Retrieved 5/9 statements.
# Partially parsed test_remove_multiple_items. Retrieved 7/11 statements.
# Partially parsed test_remove_via_delitem. Retrieved 3/7 statements.


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
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key2'
    var_5 = 'value2'
    var_6 = 'key2'
    var_7 = 'key2'

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
    var_9 = 'key3'
    var_10 = 'key2'

def test_case_0():
    var_0 = []
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = {var_1: var_2}
    var_4 = 'key1'



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 3/10 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/13 statements.
# Partially parsed test_check_and_set_attr_failed_invariant. Retrieved 3/16 statements.
# Partially parsed test_check_and_set_attr_multiple_types. Retrieved 3/10 statements.
# Partially parsed test_check_and_set_attr_no_type_checking. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = []
    var_2 = 'valid_name'
    var_3 = bool(var_1 == [])
    assert var_3 is True

def test_case_0():
    var_0 = 'name'
    var_1 = []
    var_2 = 'name'
    var_3 = 123
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'count'
    var_1 = []
    var_2 = -5
    var_3 = bool(var_1 == ['must_be_positive'])
    assert var_3 is True

def test_case_0():
    var_0 = 'value'
    var_1 = []
    var_2 = 42
    var_3 = bool(var_1 == [])
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value'
    var_2 = []
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = bool(var_2 == [])
    assert var_7 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 8/22 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 7/21 statements.
# Partially parsed test_check_and_set_attr_no_type_constraint. Retrieved 7/19 statements.
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



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
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
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
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
    var_1 = 1
    var_2 = True
    var_3 = 2
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



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_true. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_all_optional_fields. Retrieved 2/5 statements.


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
    var_3 = bool('missing_fields' in str(type(e)).lower() or 'invariant' in str(type(e)).lower())
    assert var_3 is True

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
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool('missing_fields' in str(type(e)).lower() or 'invariant' in str(type(e)).lower())
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_check_and_set_attr_valid_type_and_invariant. Retrieved 7/22 statements.
# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 7/22 statements.
# Partially parsed test_check_and_set_attr_invalid_type. Retrieved 4/19 statements.
# Partially parsed test_check_and_set_attr_multiple_valid_types. Retrieved 4/17 statements.
# Partially parsed test_check_and_set_attr_no_type_check. Retrieved 3/17 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == [])
    assert var_7 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = False
    var_2 = 'error_code_1'
    var_3 = (var_1, var_2)
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42
    var_7 = bool(var_4 == ['error_code_1'])
    assert var_7 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'string_value'
    var_4 = bool(var_1 == [])
    assert var_4 is True

def test_case_0():
    var_0 = 'TestClass'
    var_1 = []
    var_2 = 'test_field'
    var_3 = []
    var_4 = bool(var_1 == [])
    assert var_4 is True



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_none_value. Retrieved 2/5 statements.
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
    var_3 = 2

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
    var_1 = None



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_no_arguments. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 7/10 statements.


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
    var_3 = 4
    var_4 = module_0.field(initial=var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Failed to parse test_pclass_constructor_empty_class.
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
    var_0 = lambda : [1, 2, 3]
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
    var_2 = 3
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

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



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_not_allowed. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.
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
    var_1 = 1
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
    var_1 = module_0.field()
    var_2 = None
    var_3 = 2



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 8/11 statements.
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
    var_1 = '5'
    var_2 = 10

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_fails. Retrieved 3/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'test_value'
    var_3 = bool(var_0 == ['invariant_error_code'])
    assert var_3 is True



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_field_factory. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_multiple_instances_independent. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_no_arguments. Retrieved 2/5 statements.


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
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_freezes_instance. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_frozen_prevents_modification. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_field_with_factory. Retrieved 5/9 statements.


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
    var_2 = 3
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3
    var_3 = True

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_none_value. Retrieved 2/5 statements.
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



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_kwargs. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_field_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_type_checking. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_multiple_fields_and_initial. Retrieved 4/7 statements.
# Failed to parse test_pclass_new_empty_class.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 2/9 statements.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 9/13 statements.
# Partially parsed test_pclass_new_override_initial_with_kwarg. Retrieved 2/5 statements.


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
    var_4 = 'are not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'x must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {var_0: var_1}
    var_4 = module_0.pmap(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.field(initial=var_0)
    var_2 = 2
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Global invariant failed'

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_pclass_meta_new_creates_slots_with_pclass_fields. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 'test_field'
    var_3 = '__module__'
    var_4 = '__main__'
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = '__slots__'
    var_8 = '_pclass_frozen'
    var_9 = 'test_field'



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_type_check. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_field_invariant. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_preserves_field_order. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_factory_fields_parameter. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra_parameter. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_multiple_instances_independent. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_optional_field. Retrieved 6/11 statements.
# Partially parsed test_pclass_new_field_not_set_remains_missing. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 2/10 statements.
# Partially parsed test_pclass_new_with_all_fields_optional. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_passes_ignore_extra_to_factory. Retrieved 6/11 statements.


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

def test_case_0():
    var_0 = '42'

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = '42'
    var_4 = 99

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

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
    var_3 = 2
    var_4 = 'x'
    var_5 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'y'

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

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = {var_0: var_3}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_repr_format. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = 'TestPClass('
    var_5 = 'x=1'
    var_6 = "y='hello'"
    var_7 = ')'



# Parsed testcases at query #82
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



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_ignore_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_multiple_fields. Retrieved 5/8 statements.
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
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

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
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = "Can't set attribute"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 100
    var_3 = module_0.field(initial=var_2)
    var_4 = 10
    var_5 = 20



# Parsed testcases at query #84
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



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_when_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_with_invariant_errors. Retrieved 2/12 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -5
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/13 statements.
# Partially parsed test_pclass_hash_different_for_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_optional_fields. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_with_nested_values. Retrieved 7/12 statements.


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
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = (var_2, var_3)
    var_5 = 'a'
    var_6 = {var_5: var_2}



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_reduce_empty. Retrieved 1/6 statements.
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
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)

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



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_eq_predicate_isinstance_check. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_pclass_meta_weakref_not_added_when_not_pclass_bases. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = {}
    var_4 = ()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = '__weakref__'



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_frozen_after_creation. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_factory_field. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_empty_class.
# Partially parsed test_pclass_constructor_with_multiple_mandatory_fields_missing. Retrieved 2/6 statements.


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
    var_2 = 2
    var_3 = True
    var_4 = 'z'

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



