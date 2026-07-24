####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_set_new_key_marks_dirty. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_with_different_value_marks_dirty. Retrieved 4/7 statements.
# Partially parsed test_set_existing_key_with_same_value_does_not_mark_dirty. Retrieved 3/6 statements.
# Partially parsed test_set_returns_self. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = 'new_key'

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = 'key'

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'key'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_preserves_missing_fields. Retrieved 5/11 statements.


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
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = 10
    var_5 = 'y'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_without_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_missing_value. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format. Retrieved 7/11 statements.


def test_case_0():
    var_0 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = 'json'
    var_3 = 'x'
    var_4 = None
    var_5 = module_0.serialize(var_4, var_2, var_1)
    var_6 = {var_3: var_5}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_failed_invariant. Retrieved 3/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 42
    var_3 = bool(var_0 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'not_an_int'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 42
    var_3 = bool(var_0 == ['INVALID'])
    assert var_3 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_is_pclass_with_single_checkedtype_base.
# Failed to parse test_is_pclass_with_multiple_bases.
# Failed to parse test_is_pclass_with_non_checkedtype_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 5/9 statements.


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
    var_2 = 123

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'
    var_4 = 'json'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclassmeta_new_with_single_checkedtype_base. Retrieved 8/13 statements.
# Partially parsed test_pclassmeta_new_with_multiple_bases. Retrieved 5/16 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = lambda self: True
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = '__slots__'
    var_4 = '__weakref__'
    var_5 = '_pclass_frozen'
    var_6 = 'field'
    var_7 = True
    var_8 = lambda self: var_7
    var_9 = module_0.wrap_invariant(var_8)
    var_10 = (var_9,)

def test_case_0():
    var_0 = lambda self: True
    var_1 = lambda self: False
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'
    var_5 = '__weakref__'
    var_6 = '_pclass_frozen'
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field3'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_for_different_values. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_with_missing_optional_field. Retrieved 3/9 statements.
# Partially parsed test_pclass_hash_with_different_optional_field. Retrieved 4/10 statements.


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
    var_2 = 10
    var_3 = 20
    var_4 = 21

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 10
    var_4 = 20



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_eq_returns_true_for_equal_instances. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_returns_false_for_different_instances. Retrieved 5/8 statements.
# Partially parsed test_pclass_eq_returns_false_for_different_classes. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_returns_not_implemented_for_non_pclass. Retrieved 2/4 statements.


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
    var_2 = 'not_a_pclass'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_set_new_key_marks_dirty_and_adds_to_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_with_different_value_marks_dirty. Retrieved 4/7 statements.
# Partially parsed test_set_existing_key_with_same_value_does_not_mark_dirty. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'new_key'
    var_3 = 'new_value'
    var_4 = 'new_key'

def test_case_0():
    var_0 = []
    var_1 = 'existing_key'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = 'existing_key'

def test_case_0():
    var_0 = []
    var_1 = 'existing_key'
    var_2 = 'same_value'
    var_3 = {var_1: var_2}
    var_4 = 'existing_key'



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_repr_empty_pclass.
# Partially parsed test_repr_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_repr_with_missing_optional_field. Retrieved 3/6 statements.
# Partially parsed test_repr_with_string_escaping. Retrieved 2/5 statements.
# Partially parsed test_repr_with_nested_pclass. Retrieved 2/8 statements.


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
    var_5 = 2
    var_6 = 3
    var_7 = [var_3, var_5, var_6]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = "O'Reilly"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 6/10 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_4 = "'z'"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'positive'

def test_case_0():
    var_0 = 'x'
    var_1 = 'z'
    var_2 = 5
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -5
    var_3 = -3
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum_positive'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Field invariant failed'

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
    var_0 = 0
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type for field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1

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
    var_0 = lambda v: (v > 0, 'must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Global invariant failed'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestPClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 2/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'x must be positive'



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #19
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 6/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = module_0.field()
    var_5 = module_0.field()
    var_6 = 0
    var_7 = 1
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_invariant_errors_or_missing_fields.




# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/10 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'TestClass.x'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type for field TestClass.x'

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
    var_0 = 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = 21

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum must be positive'

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
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclassmeta_new_with_checkedtype_base. Retrieved 3/8 statements.
# Partially parsed test_pclassmeta_new_without_checkedtype_base. Retrieved 3/8 statements.
# Failed to parse test_pclassmeta_new_with_fields.
# Failed to parse test_pclassmeta_new_with_invariants.
# Failed to parse test_pclassmeta_new_with_inherited_fields.
# Failed to parse test_pclassmeta_new_with_inherited_invariants.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: str(v)
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v, fmt: v if fmt is None else str(v)
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'test'
    var_5 = 'custom'



# Parsed testcases at query #29
#--------------------------




def test_case_0():
    var_0 = 'error1'
    var_1 = [var_0]
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = bool(var_1 or var_3)
    assert var_4 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_equality_with_same_class_and_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 6/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 3/12 statements.
# Partially parsed test_pclass_constructor_with_global_invariant. Retrieved 4/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

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
    var_1 = -1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 4
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pclass_reduce_with_pickling. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_invariant. Retrieved 4/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 'attr'
    var_2 = 'value'
    var_3 = []



# Parsed testcases at query #38
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_invalid_field_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Field invariant failed'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type for field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

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
    var_0 = lambda v: (v > 0, 'must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Global invariant failed'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 1/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 5/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = lambda v: (v > 0, 'positive')
    var_3 = module_0.field(invariant=var_2)
    var_4 = 1
    var_5 = -1
    var_6 = 1



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_hash_returns_consistent_value. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_mixed_arguments. Retrieved 9/13 statements.
# Partially parsed test_set_with_missing_field. Retrieved 5/9 statements.
# Partially parsed test_set_with_extra_field. Retrieved 7/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10
    var_5 = 20

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
    var_6 = 'x'
    var_7 = 10
    var_8 = 20

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
    var_4 = 10
    var_5 = 30
    var_6 = 'z'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_equality_with_same_class_and_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

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
    var_0 = 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_pclass_meta_new_with_single_checked_type_base. Retrieved 3/8 statements.
# Partially parsed test_pclass_meta_new_with_multiple_bases. Retrieved 3/12 statements.
# Failed to parse test_pclass_meta_new_with_field_inheritance.
# Partially parsed test_pclass_meta_new_with_invariant_inheritance. Retrieved 4/4 statements.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'
    var_3 = '__weakref__'

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 1

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 1

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_invariant_check_fails. Retrieved 7/9 statements.


def test_case_0():
    var_0 = False
    var_1 = 'error'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'attr'
    var_6 = 'value'
    var_7 = bool(var_4 == ['error'])
    assert var_7 is True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 4/8 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/7 statements.


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
    var_2 = 'x'
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_equality_with_same_class_and_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_check_and_set_attr_with_invariant_failure. Retrieved 8/15 statements.


def test_case_0():
    var_0 = None
    var_1 = 'MockClass'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]
    var_5 = 'test_field'
    var_6 = 'test_value'
    var_7 = []
    var_8 = []
    var_9 = len(var_8)
    assert var_9 == 1
    var_10 = var_8[0]
    assert var_10 == 'Error'



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_repr_contains_class_name_and_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'TestClass'
    var_5 = 'x=1'
    var_6 = 'y=2'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 5/8 statements.
# Partially parsed test_remove_nonexistent_item_raises_attribute_error. Retrieved 4/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'a'

def test_case_0():
    var_0 = []
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = 'b'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclass_eq_same_instance. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_different_instances_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 6/10 statements.
# Partially parsed test_pclass_eq_different_classes. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_non_pclass_instance. Retrieved 2/5 statements.


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
    var_5 = 4

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_new_with_ignore_extra_and_factory. Retrieved 11/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

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
    var_1 = 1
    var_2 = 3
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0.field()
    var_3 = 5
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum_positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0.field()
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 5
    var_6 = 10
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v, ignore_extra=False: v * (3 if ignore_extra else 2)
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0.field()
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 'z'
    var_6 = 5
    var_7 = 10
    var_8 = 3
    var_9 = {var_3: var_6, var_4: var_7, var_5: var_8}
    var_10 = True
    var_11 = {var_3}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_set_with_keyword_argument. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_multiple_updates. Retrieved 8/12 statements.
# Partially parsed test_set_with_missing_field. Retrieved 7/12 statements.
# Partially parsed test_set_with_no_changes. Retrieved 4/8 statements.


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
    var_2 = 1
    var_3 = 2
    var_4 = 10
    var_5 = 30
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 4/7 statements.
# Partially parsed test_eq_different_instances_same_values. Retrieved 4/8 statements.
# Partially parsed test_eq_different_values. Retrieved 5/9 statements.
# Partially parsed test_eq_different_types. Retrieved 3/8 statements.
# Partially parsed test_eq_missing_fields. Retrieved 4/8 statements.


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
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = 2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclassmeta_new_with_single_checkedtype_base. Retrieved 5/13 statements.
# Partially parsed test_pclassmeta_new_with_multiple_bases. Retrieved 5/11 statements.
# Partially parsed test_pclassmeta_new_with_invariant_in_base. Retrieved 4/11 statements.
# Partially parsed test_pclassmeta_new_with_non_callable_invariant. Retrieved 3/7 statements.
# Partially parsed test_pclassmeta_new_with_field_inheritance. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 'TestClass'

def test_case_0():
    var_0 = lambda self: True
    var_1 = lambda self: True
    var_2 = 'c'
    var_3 = 3
    var_4 = 'TestClass'
    var_5 = 'c'
    var_6 = '__weakref__'

def test_case_0():
    var_0 = lambda self: (True, 'test')
    var_1 = {}
    var_2 = 'TestClass'
    var_3 = 0

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = 'TestClass'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'y'
    var_2 = 2
    var_3 = 'TestClass'
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 5/9 statements.
# Partially parsed test_set_with_args. Retrieved 6/10 statements.
# Partially parsed test_set_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_with_missing_field. Retrieved 5/10 statements.


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
    var_2 = 1
    var_3 = 2
    var_4 = 'y'



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_is_pclass_with_single_checked_type_base.
# Failed to parse test_is_pclass_with_multiple_bases.
# Failed to parse test_is_pclass_with_non_checked_type_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #10
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_4 = "'z'"
    var_5 = 'TestClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : 'default'
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_reduce_with_pickling. Retrieved 4/8 statements.
# Partially parsed test_pclass_reduce_with_missing_fields. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_repr_with_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_with_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_repr_with_missing_optional_field. Retrieved 3/6 statements.
# Partially parsed test_repr_with_none_value. Retrieved 2/5 statements.
# Partially parsed test_repr_with_complex_object. Retrieved 8/11 statements.


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
    var_5 = 2
    var_6 = 3
    var_7 = [var_3, var_5, var_6]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 0
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_4, var_5]
    var_7 = {var_1: var_3, var_2: var_6}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_for_different_values. Retrieved 6/12 statements.
# Partially parsed test_pclass_hash_includes_all_fields. Retrieved 7/13 statements.
# Partially parsed test_pclass_hash_with_missing_optional_field. Retrieved 4/10 statements.


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
    var_5 = 4

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = 2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 6/10 statements.
# Partially parsed test_pclass_new_with_invariant_error. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'are not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = 5

def test_case_0():
    var_0 = lambda v, ignore_extra=False: v
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (v > 0, 'must be positive')
    var_1 = module_0.field(invariant=var_0)
    var_2 = -1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = -1
    var_5 = -2
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Global invariant failed'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : 'default'
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

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
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields_raises_exception. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = lambda v: (False, 'error')
    var_3 = module_0.field(invariant=var_2)



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 1/8 statements.
# Partially parsed test_pclass_constructor_with_valid_invariant. Retrieved 1/7 statements.
# Partially parsed test_pclass_constructor_with_serializer. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_pclass_instance. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : 'default'
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = {var_2}
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Field invariant failed'

def test_case_0():
    var_0 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(serializer=var_0)
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '_pclass_fields'
    var_2 = 'x'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_equality_with_same_class_instance. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #23
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/8 statements.
# Partially parsed test_repr_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_repr_with_complex_values. Retrieved 8/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1

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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pickle_reduce_returns_tuple_with_restore_pickle_and_class_data. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_without_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_missing_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(serializer=var_0)
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v, fmt: str(v) if fmt == 'str' else v
    var_1 = module_0.field(serializer=var_0)
    var_2 = 5
    var_3 = 'str'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_preserves_other_fields. Retrieved 7/11 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/7 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'TestPClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pickle_support_returns_correct_tuple. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields_raises_exception. Retrieved 3/13 statements.


def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 1



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 4/11 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 4/12 statements.
# Partially parsed test_check_and_set_attr_with_failed_invariant. Retrieved 4/12 statements.


def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = []
    var_2 = 'attr'
    var_3 = 42
    var_4 = bool(var_1 == [])
    assert var_4 is True

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = []
    var_2 = 'attr'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda self, value: (False, 'INVALID')
    var_1 = []
    var_2 = 'attr'
    var_3 = 42
    var_4 = bool(var_1 == ['INVALID'])
    assert var_4 is True



# Parsed testcases at query #36
#--------------------------




def test_case_0():
    var_0 = 'error1'
    var_1 = [var_0]
    var_2 = []
    var_3 = bool(var_1 or var_2)
    assert var_3 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 6/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4



# Parsed testcases at query #39
#--------------------------

# Failed to parse test__is_pclass_returns_true_for_pclass_bases.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: str(v)
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v, fmt: v if fmt is None else str(v)
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 'json'



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields_raises_exception. Retrieved 1/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 1/6 statements.
# Failed to parse test_pclass_constructor_with_missing_mandatory_field.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_invalid_field_value. Retrieved 1/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'y'

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_weakref_in_slots_when_bases_are_pclass. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '__slots__'
    var_1 = ()
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'
    var_4 = '__weakref__'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 4/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = -1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 1
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: (False, 'error') if v < 0 else (True, None)
    var_1 = module_0.field(invariant=var_0)
    var_2 = 1
    var_3 = -1



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #50
#--------------------------

# Failed to parse test__is_pclass_returns_true_for_pclass_bases.




