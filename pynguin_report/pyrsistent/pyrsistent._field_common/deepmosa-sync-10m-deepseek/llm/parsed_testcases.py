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
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
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
    var_13 = module_0.check_global_invariants(var_0, var_12)
    var_14 = bool(False)
    assert var_14 is True

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
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_fields_adds_field_to_dct. Retrieved 9/13 statements.
# Partially parsed test_set_fields_handles_missing_field_in_bases. Retrieved 2/6 statements.
# Partially parsed test_set_fields_moves_pfield_instances. Retrieved 5/10 statements.
# Partially parsed test_set_fields_merges_duplicate_keys_from_bases. Retrieved 11/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_field'
    var_8 = 'a'
    var_9 = 'b'
    var_10 = 1
    var_11 = 2
    var_12 = {var_8: var_10, var_9: var_11}
    var_13 = var_6[var_7]
    var_14 = bool(var_6[var_7] == var_12)
    assert var_14 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'test_field'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = var_0[var_2]
    var_5 = bool(var_0[var_2] == {})
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = 'test_field'
    var_2 = var_0[var_1]
    var_3 = bool(var_0[var_1] == {})
    assert var_3 is True

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'normal_value'
    var_3 = ()
    var_4 = 'test_field'
    var_5 = 'key1'
    var_6 = 'key2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = ()
    var_6 = 'test_field'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = var_4['key1']
    assert var_8 == 'value1'
    var_9 = var_4['key2']
    assert var_9 == 42
    var_10 = var_4[var_6]
    var_11 = bool(var_4[var_6] == {})
    assert var_11 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'b'
    var_6 = 'c'
    var_7 = 99
    var_8 = 3
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = 'field'
    var_12 = 'a'
    var_13 = 'b'
    var_14 = 'c'
    var_15 = 1
    var_16 = 99
    var_17 = 3
    var_18 = {var_12: var_15, var_13: var_16, var_14: var_17}
    var_19 = var_10[var_11]
    var_20 = bool(var_10[var_11] == var_18)
    assert var_20 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 7/11 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 5/10 statements.
# Partially parsed test_make_seq_field_type_sets_name_using_types_to_names. Retrieved 5/11 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x > var_4
    var_6 = {}
    var_7 = 'Suffix'

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = 'cached_type'

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = '_checked_types'
    var_3 = None
    var_4 = 'Seq'

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: len(x) > var_4
    var_6 = 'Suffix'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_no_type_restriction. Retrieved 8/16 statements.
# Partially parsed test_check_type_with_type_string. Retrieved 4/11 statements.
# Partially parsed test_check_type_with_mixed_type_and_string. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field DestinationClass.test_field, was str'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'valid_string'

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'any_value'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 'test_field'
    var_4 = 42
    var_5 = 'valid_string'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'valid_string'



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_different_types.
# Failed to parse test_make_pmap_field_type_with_tuple_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 2
    var_1 = 1



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_no_type_specified. Retrieved 3/9 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_multiple_types_invalid. Retrieved 2/10 statements.
# Partially parsed test_check_type_with_type_string. Retrieved 3/9 statements.
# Partially parsed test_check_type_with_type_string_invalid. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'field_name'
    var_1 = 42

def test_case_0():
    var_0 = 'field_name'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field DestinationCls.field_name, was str'

def test_case_0():
    var_0 = None
    var_1 = 'field_name'
    var_2 = 'any_value'

def test_case_0():
    var_0 = 'field_name'
    var_1 = 42
    var_2 = 'valid_string'

def test_case_0():
    var_0 = 'field_name'
    var_1 = 3.14
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field DestinationCls.field_name, was float'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'field_name'
    var_3 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'field_name'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invalid type for field DestinationCls.field_name, was str'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_type_mismatch. Retrieved 14/20 statements.
# Partially parsed test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_type_matches. Retrieved 14/19 statements.
# Partially parsed test_check_field_parameters_initial_is_pfield_no_initial. Retrieved 13/18 statements.
# Partially parsed test_check_field_parameters_initial_not_pfield_no_initial_and_callable. Retrieved 15/20 statements.
# Partially parsed test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_empty_type. Retrieved 17/19 statements.
# Partially parsed test_check_field_parameters_initial_not_pfield_no_initial_not_callable_and_multiple_types_one_matches. Retrieved 14/19 statements.


import collections as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 123
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = bool(False)
    assert var_15 is True

import collections as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 'hello'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x

import collections as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = None
    var_12 = lambda : var_11
    var_13 = lambda x: x

import collections as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 123
    var_10 = lambda : var_9
    var_11 = True
    var_12 = lambda x: var_11
    var_13 = None
    var_14 = lambda : var_13
    var_15 = lambda x: x

import collections as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 123
    var_10 = []
    var_11 = True
    var_12 = lambda x: var_11
    var_13 = None
    var_14 = lambda : var_13
    var_15 = lambda x: x
    var_16 = var_7(initial=var_9, type=var_10, invariant=var_12, factory=var_14, serializer=var_15)
    var_17 = module_1._check_field_parameters(var_16)

import collections as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 123
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_not_type_cls. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_no_ignore_extra_param. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_ignore_extra_param. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_empty_type_tuple. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_type_set. Retrieved 2/6 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_sequence_field_creates_checked_vector_field. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_creates_checked_set_field. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_with_initial_list. Retrieved 6/10 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_optional_factory_handles_none. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_optional_factory_creates_instance. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_non_optional_factory_creates_instance. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_caches_field_type. Retrieved 3/7 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = set()
    var_2 = set()

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_2]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Failed to parse test_is_type_cls_with_type_set.
# Partially parsed test_is_type_cls_with_empty_set. Retrieved 1/2 statements.
# Failed to parse test_is_type_cls_with_single_type.
# Failed to parse test_is_type_cls_with_multiple_types.
# Partially parsed test_is_type_cls_with_non_type_first_element. Retrieved 2/3 statements.
# Failed to parse test_is_type_cls_with_subclass.
# Failed to parse test_is_type_cls_with_non_subclass.
# Partially parsed test_is_type_cls_with_string_type_name. Retrieved 2/3 statements.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/2 statements.
# Failed to parse test_is_type_cls_with_type_cls_not_type.


def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = 'not_a_type'
    var_1 = (var_0,)

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)

def test_case_0():
    var_0 = ()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 21/65 statements.


import collections as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = 'initial'
    var_2 = 'type'
    var_3 = [var_1, var_2]
    var_4 = module_0.namedtuple(var_0, var_3)
    var_5 = []
    var_6 = 5
    var_7 = lambda : var_6
    var_8 = ()
    var_9 = var_4(initial=var_6, type=var_8)
    var_10 = var_9.initial
    var_11 = var_9.initial
    var_12 = callable(var_11)
    var_13 = var_9.type
    var_14 = var_9.type
    var_15 = var_9.initial
    var_16 = var_9.initial
    var_17 = var_9.initial
    var_18 = callable(var_17)
    var_19 = var_9.type
    var_20 = var_9.type
    var_21 = var_9.initial



# Parsed testcases at query #14
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
    var_6 = (var_5, var_5)
    var_7 = lambda x: var_6
    var_8 = 200
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 500
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 600
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_an_int'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x
    var_12 = []
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #16
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
    var_0 = 'test_string'
    var_1 = 'json'

def test_case_0():
    var_0 = 123
    var_1 = 'yaml'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 12/27 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_an_int'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x
    var_12 = []



# Parsed testcases at query #18
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
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = True
    var_6 = (var_5, var_5)
    var_7 = lambda s: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = 200
    var_6 = (var_1, var_5)
    var_7 = lambda s: var_6
    var_8 = 300
    var_9 = (var_1, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_5 = False
    var_6 = 500
    var_7 = (var_5, var_6)
    var_8 = lambda s: var_7
    var_9 = 2
    var_10 = (var_1, var_9)
    var_11 = lambda s: var_10
    var_12 = False
    var_13 = 600
    var_14 = (var_12, var_13)
    var_15 = lambda s: var_14
    var_16 = [var_4, var_8, var_11, var_15]
    var_17 = module_0.check_global_invariants(var_0, var_16)
    var_18 = bool(False)
    assert var_18 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 5/9 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 4/9 statements.
# Partially parsed test_make_seq_field_type_sets_correct_name. Retrieved 4/11 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = lambda x: x > var_4

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = None

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = '_checked_types'
    var_3 = None

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_reuse_cached_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 1/7 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = None



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_field.
# Partially parsed test_pmap_field_optional_true_allows_none. Retrieved 1/10 statements.
# Failed to parse test_pmap_field_with_invariant.
# Partially parsed test_pmap_field_factory_with_optional_none. Retrieved 2/4 statements.
# Partially parsed test_pmap_field_factory_without_optional. Retrieved 3/7 statements.
# Failed to parse test_pmap_field_initial_is_checked_pmap.
# Failed to parse test_pmap_field_type_set_contains_one_element.
# Partially parsed test_pmap_field_optional_type_includes_none. Retrieved 2/7 statements.
# Partially parsed test_pmap_field_mandatory_is_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.
# Partially parsed test_pfield_constructor_with_defaults. Retrieved 5/7 statements.
# Partially parsed test_pfield_constructor_with_no_factory. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = True
    var_4 = 5
    var_5 = lambda : var_4
    var_6 = lambda x: str(x)

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = False
    var_3 = None
    var_4 = None

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = False
    var_3 = None



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 0
    var_3 = lambda x: x > var_2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = bool(False)
    assert var_3 is True

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
    var_1 = 'not_callable'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sequence_field_creates_checked_type. Retrieved 3/6 statements.
# Partially parsed test_sequence_field_with_optional. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_with_initial. Retrieved 6/9 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_factory_with_optional_none. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_factory_with_optional_value. Retrieved 4/11 statements.
# Partially parsed test_sequence_field_factory_without_optional. Retrieved 5/12 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 2
    var_3 = [var_0, var_2]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 4/11 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 5/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'Int'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_check_type_valid. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid. Retrieved 2/10 statements.
# Partially parsed test_check_type_multiple_valid. Retrieved 3/11 statements.
# Partially parsed test_check_type_no_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_type_string. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'field_name'
    var_1 = 5

def test_case_0():
    var_0 = 'field_name'
    var_1 = 'string'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field'

def test_case_0():
    var_0 = 'field_name'
    var_1 = 'string'
    var_2 = 5

def test_case_0():
    var_0 = None
    var_1 = 'field_name'
    var_2 = []

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'field_name'
    var_3 = 5



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda x: x
    var_3 = lambda x: x



# Parsed testcases at query #29
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
    var_10 = bool(False)
    assert var_10 is True

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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 5/16 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_set_fields_adds_name_dict_with_base_items. Retrieved 4/8 statements.
# Partially parsed test_set_fields_moves_pfield_instances_to_name_dict. Retrieved 3/8 statements.
# Partially parsed test_set_fields_merges_duplicate_keys_from_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_ignores_non_pfield_items_in_dct. Retrieved 5/8 statements.
# Partially parsed test_set_fields_with_mixed_base_dict_and_dct_pfields. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = {}
    var_3 = 'test_fields'
    var_4 = var_2[var_3]
    var_5 = bool(var_2[var_3] == {'field1': 'value1', 'field2': 'value2'})
    assert var_5 is True

def test_case_0():
    var_0 = 'custom_field'
    var_1 = ()
    var_2 = 'fields'
    var_3 = 'custom_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = var_0[var_2]
    var_5 = bool(var_0[var_2] == {})
    assert var_5 is True

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = {}
    var_3 = 'merged'
    var_4 = var_2[var_3]
    var_5 = bool(var_2[var_3] == {'key': 'value1', 'key': 'value2'})
    assert var_5 is True

def test_case_0():
    var_0 = 'regular_field'
    var_1 = 'pfield'
    var_2 = 'regular_value'
    var_3 = ()
    var_4 = 'fields'
    var_5 = 'pfield'

def test_case_0():
    var_0 = 'base_value'
    var_1 = 'dct_field'
    var_2 = 'collected'
    var_3 = 'dct_field'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_set_fields_pfield_condition_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'normal'
    var_3 = []
    var_4 = 'meta'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_set_fields_pfield_condition_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'not_pfield'
    var_3 = []
    var_4 = 'test_name'



# Parsed testcases at query #34
#--------------------------






# Parsed testcases at query #35
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 12/20 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_an_int'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x
    var_12 = []
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #36
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
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = True
    var_6 = (var_5, var_1)
    var_7 = lambda s: var_6
    var_8 = 300
    var_9 = (var_1, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 12/16 statements.


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
    var_8 = 'create'
    var_9 = lambda self, data, _factory_fields: data
    var_10 = {var_8: var_9}
    var_11 = [var_6, var_7, var_10]
    var_12 = module_0._restore_seq_field_pickle(var_0, var_1, var_5)
    var_13 = bool(var_12 == var_5)
    assert var_13 is True



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_caches_and_returns_cached.
# Failed to parse test_make_pmap_field_type_with_tuple_types.




# Parsed testcases at query #39
#--------------------------

# Partially parsed test_sequence_field_creates_checked_vector_field. Retrieved 6/10 statements.
# Partially parsed test_sequence_field_creates_checked_set_field. Retrieved 5/9 statements.
# Partially parsed test_sequence_field_with_optional_true. Retrieved 2/7 statements.
# Partially parsed test_sequence_field_with_custom_invariant. Retrieved 3/8 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_factory_handles_none_for_optional. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_factory_creates_instance_for_non_optional. Retrieved 6/12 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = [var_1, var_2]

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]
    var_3 = [var_1]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #40
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



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_an_int'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = 0
    var_3 = lambda x: x > var_2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = bool(False)
    assert var_3 is True

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
    var_1 = 'not_callable'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.
# Partially parsed test_pfield_constructor_with_defaults. Retrieved 5/7 statements.
# Partially parsed test_pfield_constructor_with_pfield_no_factory. Retrieved 4/8 statements.


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
    var_1 = None
    var_2 = False
    var_3 = None
    var_4 = None

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = False
    var_3 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_no_type_restriction. Retrieved 8/16 statements.
# Partially parsed test_check_type_with_type_string. Retrieved 4/11 statements.
# Partially parsed test_check_type_with_mixed_type_and_string. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 'field_name'
    var_1 = 42

def test_case_0():
    var_0 = 'field_name'
    var_1 = 'not_an_int'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field'

def test_case_0():
    var_0 = 'field_name'
    var_1 = 42
    var_2 = 'string'

def test_case_0():
    var_0 = None
    var_1 = 'field_name'
    var_2 = 42
    var_3 = 'string'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = (var_0, var_1)
    var_3 = 'field_name'
    var_4 = 42
    var_5 = 'string'

def test_case_0():
    var_0 = 'field_name'
    var_1 = 42
    var_2 = 'string'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_callable_initial_valid. Retrieved 2/5 statements.
# Partially parsed test_check_field_parameters_initial_matches_type. Retrieved 1/4 statements.
# Partially parsed test_check_field_parameters_initial_matches_one_of_multiple_types. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_initial_callable_with_type. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0._check_field_parameters(var_0)

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 2/10 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 3/10 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 2/8 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'Int'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_sequence_field_creates_checked_type_with_item_type. Retrieved 3/11 statements.
# Partially parsed test_sequence_field_optional_none_allowed. Retrieved 2/6 statements.
# Partially parsed test_sequence_field_optional_factory_creates_instance. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_non_optional_factory_creates_instance. Retrieved 7/13 statements.
# Partially parsed test_sequence_field_invariant_applied. Retrieved 5/14 statements.
# Partially parsed test_sequence_field_item_invariant_applied. Retrieved 4/10 statements.
# Partially parsed test_sequence_field_initial_empty. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_initial_with_values. Retrieved 4/9 statements.
# Partially parsed test_sequence_field_mandatory_true. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_type_set_correctly. Retrieved 2/8 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = 'a'
    var_3 = 'b'
    var_4 = [var_2, var_3]

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
    var_3 = []
    var_4 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = [var_1]
    var_3 = [var_1]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = 1.0
    var_2 = 2.0
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #6
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
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 300
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
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



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 2/6 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 2/7 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = 123
    var_1 = 5
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 10
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = ()
    var_1 = 5
    var_2 = True
    var_3 = lambda x: var_2

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #8
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
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = [var_4]
    var_6 = module_0.check_global_invariants(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 200
    var_3 = (var_1, var_2)
    var_4 = lambda s: var_3
    var_5 = True
    var_6 = (var_5, var_1)
    var_7 = lambda s: var_6
    var_8 = 300
    var_9 = (var_1, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 'not_an_int'
    var_8 = True
    var_9 = lambda x: var_8
    var_10 = lambda x: x
    var_11 = lambda x: x
    var_12 = bool(False)
    assert var_12 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_with_key_and_value_types.
# Partially parsed test_pmap_field_optional_true_allows_none. Retrieved 2/10 statements.
# Failed to parse test_pmap_field_invariant_is_wrapped.
# Partially parsed test_pmap_field_without_optional_does_not_allow_none. Retrieved 1/6 statements.
# Failed to parse test_pmap_field_initial_is_empty_pmap.
# Failed to parse test_pmap_field_mandatory_is_true.
# Partially parsed test_pmap_field_optional_factory_handles_none. Retrieved 2/4 statements.
# Partially parsed test_pmap_field_optional_factory_creates_pmap. Retrieved 3/6 statements.
# Partially parsed test_pmap_field_non_optional_factory_creates_pmap. Retrieved 4/7 statements.
# Failed to parse test_pmap_field_serializer_default.
# Failed to parse test_pmap_field_check_field_parameters.
# Failed to parse test_pmap_field_with_custom_invariant.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False

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



# Parsed testcases at query #11
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
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = bool(False)
    assert var_3 is True

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
    var_1 = 'not_callable'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = 0
    var_2 = lambda x: x > var_1
    var_3 = 'not_callable'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda x: True
    var_2 = lambda x: x
    var_3 = lambda x: x



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 5/16 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 2/5 statements.
# Failed to parse test_check_field_parameters_no_initial.
# Partially parsed test_check_field_parameters_initial_matches_type. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 42
    var_1 = lambda : var_0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0._check_field_parameters(var_1)

def test_case_0():
    var_0 = 100



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 1/10 statements.


def test_case_0():
    var_0 = True



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
    var_10 = bool(False)
    assert var_10 is True

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

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 2/9 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_none_value_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_none_value_and_custom_serializer. Retrieved 2/5 statements.


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
    var_1 = 'csv'

def test_case_0():
    var_0 = 'json'
    var_1 = None

def test_case_0():
    var_0 = 'xml'
    var_1 = None



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_restore_pmap_field_pickle.




# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pmap_field_factory_optional_none. Retrieved 2/13 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_global_invariants_subject_passed_correctly. Retrieved 2/8 statements.


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
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = (var_5, var_5)
    var_7 = lambda x: var_6
    var_8 = 3
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

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
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = None
    var_1 = 'specific_subject'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 6/30 statements.


def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x
    var_6 = 0



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_checked_type_with_no_serializer_different_format. Retrieved 2/7 statements.
# Partially parsed test_serialize_non_checked_type_with_no_serializer. Retrieved 4/5 statements.
# Partially parsed test_serialize_checked_type_with_serializer. Retrieved 2/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []

def test_case_0():
    var_0 = None
    var_1 = 'xml'
    var_2 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = []
    var_1 = lambda fmt, val: f'serialized_{fmt}_{val}'
    var_2 = 'json'
    var_3 = 'test_value'
    var_4 = module_0.serialize(var_1, var_2, var_3)
    assert var_4 == 'serialized_json_test_value'

def test_case_0():
    var_0 = lambda fmt, val: f'custom_{fmt}_{val.__class__.__name__}'
    var_1 = 'json'
    var_2 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda fmt, val: f'serialized_{fmt}_{val}'
    var_1 = 'yaml'
    var_2 = 42
    var_3 = module_0.serialize(var_0, var_1, var_2)
    assert var_3 == 'serialized_yaml_42'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_false_when_ignore_extra_false. Retrieved 1/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_when_not_type_cls. Retrieved 1/7 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_when_no_ignore_extra_param. Retrieved 1/12 statements.
# Partially parsed test_is_field_ignore_extra_complaint_true_when_all_conditions_met. Retrieved 2/14 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = 'ignore_extra'
    var_1 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 2/9 statements.
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



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.
# Partially parsed test_pfield_constructor_with_defaults. Retrieved 5/7 statements.
# Partially parsed test_pfield_constructor_with_multiple_types. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = True
    var_4 = lambda x: x
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = False
    var_3 = None
    var_4 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 0
    var_4 = True
    var_5 = lambda x: x
    var_6 = lambda x: x
    var_7 = module_0._PField(var_0, var_2, var_3, var_4, var_5, var_6)
    var_8 = var_7.type
    var_9 = bool(var_7.type == var_0)
    assert var_9 is True
    var_10 = var_7.invariant
    var_11 = bool(var_7.invariant == var_2)
    assert var_11 is True
    var_12 = var_7.initial
    var_13 = bool(var_7.initial == var_3)
    assert var_13 is True
    var_14 = var_7.mandatory
    var_15 = bool(var_7.mandatory == var_4)
    assert var_15 is True
    var_16 = var_7._factory
    var_17 = bool(var_7._factory == var_5)
    assert var_17 is True
    var_18 = var_7.serializer
    var_19 = bool(var_7.serializer == var_6)
    assert var_19 is True

def test_case_0():
    var_0 = 0.0
    var_1 = False
    var_2 = lambda x: float(x)
    var_3 = lambda x: repr(x)



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_is_type_cls_with_type_set.
# Partially parsed test_is_type_cls_with_empty_set. Retrieved 1/2 statements.
# Failed to parse test_is_type_cls_with_single_type_tuple.
# Failed to parse test_is_type_cls_with_multiple_type_tuple.
# Failed to parse test_is_type_cls_with_non_matching_type_cls.
# Failed to parse test_is_type_cls_with_matching_subclass.
# Partially parsed test_is_type_cls_with_string_type_name. Retrieved 2/3 statements.
# Partially parsed test_is_type_cls_with_string_type_name_and_subclass. Retrieved 2/3 statements.
# Partially parsed test_is_type_cls_with_mixed_string_and_type. Retrieved 1/3 statements.
# Partially parsed test_is_type_cls_with_empty_tuple. Retrieved 1/2 statements.


def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)

def test_case_0():
    var_0 = 'builtins.int'

def test_case_0():
    var_0 = ()



# Parsed testcases at query #28
#--------------------------

# Failed to parse test_restore_pmap_field_pickle.




# Parsed testcases at query #29
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_error_codes_present. Retrieved 3/22 statements.


def test_case_0():
    var_0 = 'test_subject'
    var_1 = ()
    var_2 = 'Global invariant failed'
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 5/13 statements.
# Partially parsed test_sequence_field_with_optional_false. Retrieved 4/9 statements.
# Partially parsed test_sequence_field_with_none_initial_and_optional_true. Retrieved 2/9 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 5/9 statements.
# Partially parsed test_sequence_field_with_invariant. Retrieved 6/12 statements.


def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 0
    var_3 = 'positive'
    var_4 = lambda x: (x > var_2, var_3)

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = None
    var_3 = 0
    var_4 = 'non-empty'
    var_5 = lambda x: (x is var_2 or len(x) > var_3, var_4)



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_caches_result. Retrieved 2/10 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 3/10 statements.
# Partially parsed test_make_seq_field_type_has_reduce. Retrieved 2/8 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'Int'

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_restore_seq_field_pickle.




# Parsed testcases at query #33
#--------------------------






# Parsed testcases at query #34
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 6/21 statements.


def test_case_0():
    var_0 = []
    var_1 = 'not_an_int'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = lambda : var_4
    var_6 = lambda x: x
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'Initial has invalid type'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_different_types.
# Failed to parse test_make_pmap_field_type_with_tuple_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_global_invariants_subject_passed_correctly. Retrieved 2/8 statements.


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
    var_1 = True
    var_2 = 0
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 2
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 3
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = True
    var_9 = (var_8, var_1)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 6
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = None
    assert var_0 == 'specific'
    var_1 = 'specific'



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_restore_seq_field_pickle.




# Parsed testcases at query #38
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'json'



