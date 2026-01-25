####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 200
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 20
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 30
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = 2
    var_8 = (var_1, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_4, var_6, var_9]
    var_11 = module_0.check_global_invariants(var_0, var_10)



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_custom_class_name.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1.0
    var_1 = True
    var_2 = {var_0: var_1}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_sequence_field_creates_checked_type_with_item_type. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_optional_handles_none. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_optional_creates_instance. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_non_optional_creates_instance. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_initial_value. Retrieved 4/9 statements.
# Partially parsed test_sequence_field_invariant. Retrieved 5/14 statements.
# Partially parsed test_sequence_field_item_invariant. Retrieved 7/16 statements.
# Partially parsed test_sequence_field_mandatory_true. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_type_caching. Retrieved 3/10 statements.


def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 5
    var_2 = 10
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 1
    var_3 = [var_2]
    var_4 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = -1
    var_6 = [var_5]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 1/10 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_callable_initial_allowed. Retrieved 2/5 statements.
# Failed to parse test_check_field_parameters_no_initial_allowed.
# Partially parsed test_check_field_parameters_valid_initial_in_type_list. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 'default'
    var_1 = lambda : var_0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0._check_field_parameters(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0._check_field_parameters(var_1)

def test_case_0():
    var_0 = 123



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 1/14 statements.


def test_case_0():
    var_0 = {}



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_type.
# Partially parsed test_pmap_field_optional_true_allows_none. Retrieved 2/9 statements.
# Failed to parse test_pmap_field_invariant_passed_through.
# Partially parsed test_pmap_field_factory_creates_checked_pmap. Retrieved 5/11 statements.
# Partially parsed test_pmap_field_optional_factory_handles_none. Retrieved 4/11 statements.
# Failed to parse test_pmap_field_type_set_includes_key_and_value_types.
# Partially parsed test_pmap_field_optional_type_includes_none. Retrieved 2/6 statements.
# Failed to parse test_pmap_field_initial_is_empty_checked_pmap.
# Failed to parse test_pmap_field_mandatory_is_true.
# Failed to parse test_pmap_field_without_invariant_uses_default.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_0: var_2}

def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_types_to_names_with_builtin_types.
# Failed to parse test_types_to_names_with_single_type.
# Partially parsed test_types_to_names_with_mixed_types. Retrieved 1/3 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'collections.abc.Sequence'
    var_1 = 'typing.Optional'
    var_2 = (var_0, var_1)
    var_3 = module_0._types_to_names(var_2)
    assert var_3 == 'SequenceOptional'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._types_to_names(var_0)
    assert var_1 == ''

def test_case_0():
    var_0 = 'typing.Dict'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = True
    var_4 = 5
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_restore_seq_field_pickle.




# Parsed testcases at query #11
#--------------------------

# Failed to parse test_field_with_single_type.
# Failed to parse test_field_with_multiple_types_as_list.
# Failed to parse test_field_with_multiple_types_as_set.
# Failed to parse test_field_with_multiple_types_as_tuple.
# Partially parsed test_field_with_initial_value. Retrieved 1/2 statements.
# Partially parsed test_field_invalid_initial_type_raises. Retrieved 1/3 statements.
# Failed to parse test_field_with_nested_iterable_types.
# Failed to parse test_field_with_preserved_iterable_type.
# Partially parsed test_field_invariant_wrapping. Retrieved 10/11 statements.
# Partially parsed test_field_invariant_single_bool_result. Retrieved 6/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.field(var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.field(invariant=var_3)
    var_5 = var_4.invariant
    var_6 = callable(var_5)

def test_case_0():
    var_0 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = module_0.field(factory=var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = module_0.field(serializer=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)

def test_case_0():
    var_0 = 'not_int'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(invariant=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(factory=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(serializer=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = lambda x: var_6
    var_8 = module_0.field(invariant=var_7)
    var_9 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.field(invariant=var_3)
    var_5 = None



# Parsed testcases at query #12
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 100
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 10
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = True
    var_9 = 15
    var_10 = (var_8, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_7, var_11]
    var_13 = module_0.check_global_invariants(var_0, var_12)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = False
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 2
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 3
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_field_parameters_predicate_at_line_3_false. Retrieved 3/18 statements.


def test_case_0():
    var_0 = 'int'
    var_1 = [var_0]
    var_2 = 'str'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_factory_property_returns_type_create_when_no_factory_and_single_checkedtype. Retrieved 3/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = None



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_different_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #16
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 101
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = (var_4, var_0)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 0
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = [var_3, var_5]
    var_7 = 'test_subject'
    var_8 = module_0.check_global_invariants(var_7, var_6)



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 5/10 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 5/10 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 5/10 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

def test_case_0():
    var_0 = ()
    var_1 = 'anything'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = 'not_callable'
    var_2 = lambda x: x
    var_3 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'
    var_4 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = 'not_callable'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 2/9 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'custom_xml_{}'

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'yaml'

def test_case_0():
    var_0 = 123
    var_1 = 'json'



# Parsed testcases at query #20
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 101
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = (var_4, var_0)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)



# Parsed testcases at query #21
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 'ERROR_001'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = True
    var_5 = 'ERROR_002'
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_3, var_7]
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = 'ERROR_001'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = 'ERROR_002'
    var_5 = (var_0, var_4)
    var_6 = lambda x: var_5
    var_7 = [var_3, var_6]
    var_8 = 'test_subject'
    var_9 = module_0.check_global_invariants(var_8, var_7)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_not_type_cls. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_no_ignore_extra_param. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_has_ignore_extra_param. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_empty_type_tuple. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_type_as_set. Retrieved 2/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = set()
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = ()
    var_1 = None
    var_2 = True

def test_case_0():
    var_0 = None
    var_1 = True



# Parsed testcases at query #23
#--------------------------




import builtins as module_0
import pyrsistent._checked_types as module_1
import pyrsistent._field_common as module_2

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_1.CheckedType()
    var_2 = 'json'
    var_3 = module_2.serialize(var_0, var_2, var_1)
    assert var_3 == 'serialized_json'



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_restore_seq_field_pickle.




# Parsed testcases at query #25
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 2/11 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 5/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_custom_class_name.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 1.0
    var_1 = True
    var_2 = {var_0: var_1}
    var_3 = 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 3/18 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 0



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_restore_seq_field_pickle.




# Parsed testcases at query #30
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda s: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = False
    var_6 = 100
    var_7 = (var_5, var_6)
    var_8 = lambda s: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = 300
    var_6 = (var_1, var_5)
    var_7 = lambda s: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda s: var_5
    var_7 = 2
    var_8 = (var_1, var_7)
    var_9 = lambda s: var_8
    var_10 = [var_4, var_6, var_9]
    var_11 = module_0.check_global_invariants(var_0, var_10)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 400
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = True
    var_6 = (var_5, var_1)
    var_7 = lambda s: var_6
    var_8 = 500
    var_9 = (var_1, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_field_factory_for_optional_none. Retrieved 4/19 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 6/10 statements.


import builtins as module_0
import pyrsistent._field_common as module_1
import pyrsistent._checked_types as module_2

def test_case_0():
    var_0 = module_0.object()
    var_1 = module_0.object()
    var_2 = module_0.object()
    var_3 = module_0.object()
    var_4 = module_1._restore_seq_field_pickle(var_0, var_1, var_2)
    var_5 = module_2._restore_pickle(var_3, var_2)



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_type_cls_check_fails. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_field_initial_not_callable_and_type_mismatch. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'initial'
    var_3 = 'type'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 42
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_set_fields_with_base_dict. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_base_dict_and_pfield. Retrieved 5/10 statements.
# Partially parsed test_set_fields_with_overlapping_keys_in_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_empty_base_dict_and_pfield. Retrieved 5/10 statements.
# Partially parsed test_set_fields_with_no_name_in_base. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test'
    var_3 = module_0.set_fields(var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'test'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = 'key'
    var_2 = {var_1: var_0}
    var_3 = []
    var_4 = 'test'
    var_5 = module_0.set_fields(var_2, var_3, var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'base_key'
    var_1 = 'base_value'
    var_2 = {var_0: var_1}
    var_3 = module_0._PField()
    var_4 = 'pkey'
    var_5 = {var_4: var_3}
    var_6 = 'test'

def test_case_0():
    var_0 = 'a'
    var_1 = 'c'
    var_2 = 1
    var_3 = 3
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 2
    var_8 = 4
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0._PField()
    var_2 = 'key'
    var_3 = {var_2: var_1}
    var_4 = 'test'

def test_case_0():
    var_0 = {}
    var_1 = 'test'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = 'xml'

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'json'

def test_case_0():
    var_0 = 'test_value'
    var_1 = 'xml'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_callable_initial_allowed. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 3/6 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_initial_matches_type. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_initial_matches_one_of_multiple_types. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = ()
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 5
    var_1 = 'not_callable'

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'

def test_case_0():
    var_0 = 'hello'
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serialize_checked_type_with_other_serializer. Retrieved 2/8 statements.
# Partially parsed test_serialize_non_checked_type_with_pfield_no_serializer. Retrieved 5/11 statements.
# Partially parsed test_serialize_non_checked_type_with_custom_serializer. Retrieved 2/5 statements.


import builtins as module_0
import pyrsistent._checked_types as module_1
import pyrsistent._field_common as module_2

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'json'
    var_2 = module_1.CheckedType()
    var_3 = module_2.serialize(var_0, var_1, var_2)
    assert var_3 == 'serialized_json'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = module_0.CheckedType()

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = lambda s, f, v: default_serializer(f, v) if s is var_0 else s(f, v)
    var_2 = 'json'
    var_3 = 42
    var_4 = module_1.serialize(var_0, var_2, var_3)
    assert var_4 == 'default_json_42'

def test_case_0():
    var_0 = 'xml'
    var_1 = 100



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 6/11 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_callable_initial_allowed. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_no_initial_allowed. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = 'default'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 'default'
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = 'default'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x
    var_6 = module_0.field(var_0, var_3, var_1, factory=var_4, serializer=var_5)
    var_7 = module_0._check_field_parameters(var_6)

def test_case_0():
    var_0 = 'default'
    var_1 = 'not_callable'
    var_2 = lambda x: x
    var_3 = lambda x: x

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'
    var_4 = lambda x: x

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = 'not_callable'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 3/18 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 0



# Parsed testcases at query #41
#--------------------------






# Parsed testcases at query #42
#--------------------------

# Partially parsed test_factory_property_when_factory_is_not_pfield_no_factory. Retrieved 7/11 statements.
# Partially parsed test_factory_property_when_type_length_not_one. Retrieved 7/11 statements.
# Partially parsed test_factory_property_when_type_element_not_checkedtype_subclass. Retrieved 4/11 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = True
    var_4 = 42
    var_5 = lambda : var_4
    var_6 = None

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 0
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = True
    var_6 = None

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = True
    var_3 = None



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_factory_property_with_non_checked_type. Retrieved 3/12 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = None
    var_2 = True



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = 42
    var_5 = 'test'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 2/10 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'custom_xml_{}'

def test_case_0():
    var_0 = 'test_string'
    var_1 = 'json'

def test_case_0():
    var_0 = 123
    var_1 = 'yaml'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__restore_seq_field_pickle. Retrieved 10/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'TestItem'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'MockType'
    var_7 = ()
    var_8 = {}
    var_9 = module_0._restore_seq_field_pickle(var_0, var_1, var_5)



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_different_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_valid_initial_in_type. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123

def test_case_0():
    var_0 = 123

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0._check_field_parameters(var_1)

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x

def test_case_0():
    var_0 = 'default'
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = module_0.field(invariant=var_2, initial=var_0, factory=var_3, serializer=var_4)
    var_6 = module_0._check_field_parameters(var_5)

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 2/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_not_type_cls. Retrieved 2/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_no_ignore_extra_param. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_has_ignore_extra_param. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_empty_types. Retrieved 2/5 statements.
# Partially parsed test_is_field_ignore_extra_complaint_set_type. Retrieved 1/5 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = False

def test_case_0():
    var_0 = set()
    var_1 = True

def test_case_0():
    var_0 = lambda x: x
    var_1 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = ()
    var_1 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sequence_field_with_optional_false. Retrieved 5/12 statements.
# Partially parsed test_sequence_field_with_optional_true. Retrieved 5/14 statements.
# Partially parsed test_sequence_field_with_none_initial_when_optional. Retrieved 2/10 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 6/16 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 5/14 statements.
# Partially parsed test_sequence_field_factory_with_optional_true_and_none. Retrieved 3/10 statements.
# Partially parsed test_sequence_field_factory_with_optional_true_and_value. Retrieved 9/18 statements.
# Partially parsed test_sequence_field_factory_with_optional_false. Retrieved 9/18 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_checked_type_with_other_serializer. Retrieved 2/8 statements.
# Partially parsed test_serialize_non_checked_type_with_pfield_no_serializer. Retrieved 4/5 statements.
# Partially parsed test_serialize_non_checked_type_with_custom_serializer. Retrieved 2/5 statements.


import builtins as module_0
import pyrsistent._checked_types as module_1
import pyrsistent._field_common as module_2

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'json'
    var_2 = module_1.CheckedType()
    var_3 = module_2.serialize(var_0, var_1, var_2)
    assert var_3 == 'serialized_json'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'json'
    var_1 = module_0.CheckedType()

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'json'
    var_2 = 'test_value'
    var_3 = module_1.serialize(var_0, var_1, var_2)

def test_case_0():
    var_0 = 'json'
    var_1 = 'test'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_callable_initial_allowed. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_type_no_initial_check. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_initial_pfield_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_initial_matches_type. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = ()
    var_1 = 'anything'
    var_2 = True
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = 5
    var_1 = 'not_callable'

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 'hello'
    var_1 = True
    var_2 = lambda x: var_1



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 10
    var_3 = True
    var_4 = 5
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_caches_classes.
# Failed to parse test_make_pmap_field_type_with_custom_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_no_type_specified. Retrieved 3/9 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_multiple_types_invalid. Retrieved 2/10 statements.
# Partially parsed test_check_type_with_type_string. Retrieved 3/9 statements.
# Partially parsed test_check_type_with_type_string_invalid. Retrieved 3/10 statements.
# Partially parsed test_check_type_empty_type_tuple. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any_value'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'valid_string'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 3.14

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 'not_an_int'

def test_case_0():
    var_0 = ()
    var_1 = 'test_field'
    var_2 = 42



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_sets_name_using_types_to_names. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 2/9 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = None
    var_1 = lambda x: x is not var_0

def test_case_0():
    var_0 = None
    var_1 = 'SEQ_FIELD_TYPE_SUFFIXES'

def test_case_0():
    var_0 = 0
    var_1 = lambda d: len(d) > var_0



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_optional_field_factory_returns_none_when_argument_is_none. Retrieved 10/17 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = False
    var_4 = 'MockCheckedType'
    var_5 = ()
    var_6 = 'create'
    var_7 = 'created'
    var_8 = lambda self, *args, **kwargs: var_7
    var_9 = {var_6: var_8}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_initial_not_callable_and_not_instance_of_type. Retrieved 12/18 statements.
# Partially parsed test_initial_not_callable_and_type_empty. Retrieved 14/17 statements.
# Partially parsed test_initial_is_callable. Retrieved 13/18 statements.
# Partially parsed test_initial_is_PFIELD_NO_INITIAL. Retrieved 12/17 statements.
# Partially parsed test_initial_is_instance_of_type. Retrieved 12/17 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'initial'
    var_3 = 'type'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 42
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'initial'
    var_3 = 'type'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 42
    var_8 = ()
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = lambda x: x
    var_12 = lambda x: x
    var_13 = {var_2: var_7, var_3: var_8, var_4: var_10, var_5: var_11, var_6: var_12}

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'initial'
    var_3 = 'type'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 42
    var_8 = lambda : var_7
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = lambda x: x
    var_12 = lambda x: x

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'Field'
    var_2 = ()
    var_3 = 'initial'
    var_4 = 'type'
    var_5 = 'invariant'
    var_6 = 'factory'
    var_7 = 'serializer'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'initial'
    var_3 = 'type'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'hello'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_check_global_invariants_subject_passed_to_invariants. Retrieved 2/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda s: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = False
    var_6 = 100
    var_7 = (var_5, var_6)
    var_8 = lambda s: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = 300
    var_6 = (var_1, var_5)
    var_7 = lambda s: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 400
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = 500
    var_6 = (var_1, var_5)
    var_7 = lambda s: var_6
    var_8 = 600
    var_9 = (var_1, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = None
    var_1 = 'specific_subject'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_factory_property_with_checked_type. Retrieved 2/10 statements.


def test_case_0():
    var_0 = None
    var_1 = True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 0
    var_3 = lambda x: x > var_2

def test_case_0():
    var_0 = 'string'
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = lambda x: x > var_2

def test_case_0():
    var_0 = ()
    var_1 = 5
    var_2 = 0
    var_3 = lambda x: x > var_2

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not callable'

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not callable'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 0
    var_3 = lambda x: x > var_2

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0
    var_2 = 0
    var_3 = lambda x: x > var_2

def test_case_0():
    var_0 = ()
    var_1 = 'anything'
    var_2 = True
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = 5
    var_1 = 'not_callable'

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_pfield_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_with_checked_type_and_other_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_non_checked_type_and_pfield_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = 'xml'

def test_case_0():
    var_0 = 'some_value'
    var_1 = 'json'

def test_case_0():
    var_0 = 123
    var_1 = 'yaml'



# Parsed testcases at query #20
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = module_0.check_global_invariants(var_0, var_5)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 201
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 202
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0.check_global_invariants(var_0, var_12)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 300
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 301
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 302
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 10/22 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 7/16 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 7/16 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 6/15 statements.
# Partially parsed test_sequence_field_initial_empty. Retrieved 4/11 statements.


def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 5
    var_7 = 6
    var_8 = [var_5, var_6, var_7]
    var_9 = None

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'c'
    var_5 = 'd'
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 3
    var_5 = 4
    var_6 = [var_4, var_5]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]
    var_3 = 2
    var_4 = 3
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 1
    var_3 = [var_2]



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.
# Partially parsed test_pfield_constructor_with_defaults. Retrieved 4/7 statements.
# Partially parsed test_pfield_constructor_with_no_factory. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = True
    var_4 = 10
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)

def test_case_0():
    var_0 = None
    var_1 = 'default'
    var_2 = False
    var_3 = None

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = True
    var_3 = lambda x: x



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 2/17 statements.


def test_case_0():
    var_0 = True
    var_1 = 0



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_custom_class_name.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1.0
    var_1 = True
    var_2 = {var_0: var_1}



# Parsed testcases at query #26
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 200
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 20
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 30
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = (var_1, var_1)
    var_3 = lambda x: var_2
    var_4 = 2
    var_5 = (var_1, var_4)
    var_6 = lambda x: var_5
    var_7 = 3
    var_8 = (var_1, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_3, var_6, var_9]
    var_11 = module_0.check_global_invariants(var_0, var_10)



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__sequence_field_creates_checked_field_with_optional_false. Retrieved 5/13 statements.
# Partially parsed test__sequence_field_creates_checked_field_with_optional_true. Retrieved 5/14 statements.
# Partially parsed test__sequence_field_with_none_initial_and_optional_true. Retrieved 2/9 statements.
# Partially parsed test__sequence_field_with_invariant. Retrieved 5/15 statements.
# Partially parsed test__sequence_field_with_item_invariant. Retrieved 5/15 statements.
# Partially parsed test__sequence_field_factory_with_optional_true_and_none_argument. Retrieved 3/11 statements.
# Partially parsed test__sequence_field_factory_with_optional_true_and_non_none_argument. Retrieved 6/16 statements.
# Partially parsed test__sequence_field_factory_with_optional_false. Retrieved 6/16 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_type.
# Partially parsed test_pmap_field_optional_true_allows_none. Retrieved 1/7 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_with_optional_true_and_none. Retrieved 2/4 statements.
# Partially parsed test_pmap_field_factory_with_optional_true_and_non_none. Retrieved 3/8 statements.
# Partially parsed test_pmap_field_factory_with_optional_false. Retrieved 4/9 statements.
# Failed to parse test_pmap_field_initial_is_empty_map.
# Failed to parse test_pmap_field_type_set_contains_single_checked_pmap_type.
# Partially parsed test_pmap_field_optional_type_includes_none. Retrieved 2/8 statements.
# Failed to parse test_pmap_field_reuses_cached_map_type.
# Failed to parse test_pmap_field_with_different_key_value_types.
# Failed to parse test_pmap_field_invariant_is_wrapped.
# Failed to parse test_pmap_field_no_invariant.
# Partially parsed test_pmap_field_mandatory_is_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 'a'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_optional_pmap_field_factory_returns_none_when_argument_is_none. Retrieved 9/11 statements.


def test_case_0():
    var_0 = True
    var_1 = 'CheckedPMap'
    var_2 = ()
    var_3 = 'create'
    var_4 = 'map'
    var_5 = lambda x: var_4
    var_6 = {var_3: var_5}
    var_7 = None
    var_8 = lambda argument: var_7 if argument is var_7 else TheMap.create(argument)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pmap_field_factory_not_checked_type. Retrieved 1/11 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 7/12 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 5/10 statements.
# Partially parsed test_make_seq_field_type_sets_reduce_method. Retrieved 8/16 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = 0
    var_4 = lambda x: x > var_3
    var_5 = {}
    var_6 = 'Suffix'

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = None
    var_4 = 'cached_type'

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = True
    var_4 = lambda x: var_3
    var_5 = {}
    var_6 = 'List'
    var_7 = 2

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = '_checked_types'
    var_3 = None
    var_4 = {}
    var_5 = 'Vector'



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #33
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_1)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    var_8 = module_0.check_global_invariants(var_0, var_7)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 100
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 300
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = True
    var_9 = (var_8, var_1)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 400
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 500
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #35
#--------------------------






# Parsed testcases at query #36
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap.
# Partially parsed test_pmap_field_optional_true_allows_none. Retrieved 4/9 statements.
# Failed to parse test_pmap_field_invariant_passed_through.
# Failed to parse test_pmap_field_initial_is_empty_map.
# Partially parsed test_pmap_field_factory_creates_checked_map. Retrieved 5/8 statements.
# Partially parsed test_pmap_field_with_optional_false_does_not_allow_none. Retrieved 2/4 statements.
# Partially parsed test_pmap_field_key_and_value_types_respected. Retrieved 3/7 statements.
# Failed to parse test_pmap_field_mandatory_is_true.
# Failed to parse test_pmap_field_serializer_default.
# Failed to parse test_pmap_field_invariant_default.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_0: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = False
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = {var_0: var_1}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda x: True
    var_2 = lambda x: x
    var_3 = lambda x: x



