####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_invariant_errors_empty_invariants. Retrieved 2/4 statements.
# Partially parsed test_invariant_errors_passes_elem_to_invariants. Retrieved 2/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = []
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)
    var_14 = [var_6, var_9]
    var_15 = bool(var_13 == var_14)
    assert var_15 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'err2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = [var_2, var_5]
    var_11 = bool(var_9 == var_10)
    assert var_11 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 42
    var_2 = bool(var_0 == [var_1])
    assert var_2 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_creates_empty_checkedpset. Retrieved 1/4 statements.
# Partially parsed test_constructor_accepts_iterable_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_enforces_type_constraint. Retrieved 3/7 statements.
# Partially parsed test_constructor_enforces_invariant_constraint. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 6/9 statements.
# Partially parsed test_constructor_duplicates_removed. Retrieved 3/6 statements.
# Partially parsed test_constructor_repr_custom_class_name. Retrieved 4/9 statements.
# Partially parsed test_constructor_str_matches_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_serialize_produces_set. Retrieved 4/8 statements.
# Partially parsed test_constructor_pickle_support. Retrieved 4/10 statements.
# Partially parsed test_constructor_evolver_returns_evolver_instance. Retrieved 4/10 statements.
# Partially parsed test_constructor_create_classmethod. Retrieved 4/7 statements.
# Partially parsed test_constructor_inheritance_type_check. Retrieved 3/8 statements.
# Partially parsed test_constructor_empty_invariant. Retrieved 4/7 statements.
# Partially parsed test_constructor_multiple_types. Retrieved 3/6 statements.


def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -1
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_0, var_1, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'TestSet'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = lambda format, v: str(v)
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = -1
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Failed to parse test_maybe_parse_user_type_with_list_of_types.
# Failed to parse test_maybe_parse_user_type_with_complex_nested_iterable.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'str'
    var_1 = 'int'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('str', 'int'))
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/9 statements.
# Partially parsed test_constructor_multiple_invariants. Retrieved 7/14 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 1
    var_8 = 0.0
    var_9 = {var_7: var_8}
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test__restore_pickle. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = set()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 3/7 statements.
# Partially parsed test_check_types_with_no_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 2/8 statements.
# Partially parsed test_check_types_with_type_string. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_mixed_type_and_string. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_type_string. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'hello'
    var_3 = 2.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'invalid'
    var_1 = [var_0]
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 'hello'
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 1
    var_2 = 'hello'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'invalid'
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 10/11 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_empty_input. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_mixed_data. Retrieved 12/13 statements.


def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = (var_0, var_3)
    var_5 = []
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = []
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_6, var_8]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = 'error2'
    var_5 = [var_4]
    var_6 = (var_0, var_5)
    var_7 = 'error3'
    var_8 = [var_7]
    var_9 = (var_0, var_8)
    var_10 = [var_3, var_6, var_9]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'err1'
    var_5 = 'err2'
    var_6 = [var_4, var_5]
    var_7 = (var_3, var_6)
    var_8 = 'err3'
    var_9 = [var_8]
    var_10 = (var_3, var_9)
    var_11 = [var_2, var_7, var_10]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_store_types_single_dict. Retrieved 3/5 statements.
# Partially parsed test_store_types_multiple_dicts. Retrieved 7/13 statements.
# Partially parsed test_store_types_iterable_source. Retrieved 3/6 statements.
# Partially parsed test_store_types_preserved_iterable. Retrieved 3/5 statements.
# Partially parsed test_store_types_source_in_base_only. Retrieved 5/9 statements.
# Partially parsed test_store_types_multiple_sources. Retrieved 6/13 statements.
# Partially parsed test_store_types_nested_iterable. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Base2'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 'MyClass'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['dest']
    var_7 = bool(var_2['dest'] == ('MyClass',))
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'dest'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0['dest']
    var_6 = bool(var_0['dest'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'source'
    var_4 = 'dest'
    var_5 = var_0['dest']

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Base2'
    var_4 = ()
    var_5 = 'dest'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'dest'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'dest'
    var_5 = 'source'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_checkedtype_constructor. Retrieved 3/12 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'
    var_2 = 'create'
    var_3 = 'serialize'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_with_default_format. Retrieved 9/13 statements.
# Partially parsed test_serialize_with_custom_format. Retrieved 8/12 statements.
# Partially parsed test_serialize_empty_set. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_complex_serializer. Retrieved 8/12 statements.


def test_case_0():
    var_0 = lambda format, v: str(v)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = {var_5, var_6, var_7}

def test_case_0():
    var_0 = lambda format, v: f'{v}:{format}'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'hex'
    var_5 = '1:hex'
    var_6 = '2:hex'
    var_7 = {var_5, var_6}

def test_case_0():
    var_0 = lambda format, v: str(v)
    var_1 = []
    var_2 = set()

def test_case_0():
    var_0 = lambda format, v: v.upper() if format == 'upper' else v
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'upper'
    var_5 = 'A'
    var_6 = 'B'
    var_7 = {var_5, var_6}



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_key_violation. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value_violation. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_satisfied. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_checked_type_creation. Retrieved 3/19 statements.
# Partially parsed test_constructor_with_checked_type_and_regular_type_mix. Retrieved 3/12 statements.
# Partially parsed test_constructor_with_checked_type_and_regular_type_mix_creation. Retrieved 5/21 statements.
# Partially parsed test_constructor_with_checked_type_no_creation_when_type_matches. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_type_value_creation. Retrieved 5/18 statements.
# Partially parsed test_constructor_with_checked_type_value_no_creation_when_type_matches. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr. Retrieved 4/10 statements.
# Partially parsed test_constructor_with_initial_as_checkedpmap_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_initial_as_checkedpmap_subclass_instance. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key must be non-negative')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key must be non-negative')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 2
    var_1 = 1.5
    var_2 = 2.5

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1.5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'raw_value'
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap({1: 1.5})'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test__checked_type_create_with_same_class. Retrieved 1/4 statements.
# Partially parsed test__checked_type_create_without_checked_types. Retrieved 5/10 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_matching_data. Retrieved 4/16 statements.
# Partially parsed test__checked_type_create_with_checked_type_and_non_matching_data. Retrieved 8/21 statements.
# Partially parsed test__checked_type_create_with_ignore_extra_true. Retrieved 7/20 statements.
# Partially parsed test__checked_type_create_with_multiple_checked_types. Retrieved 5/23 statements.
# Partially parsed test__checked_type_create_with_mixed_data_and_checked_type. Retrieved 3/16 statements.


def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = [var_4]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 3
    var_6 = 4
    var_7 = [var_5, var_6]
    var_8 = [var_4, var_7]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = [var_4]
    var_6 = True
    var_7 = 0

def test_case_0():
    var_0 = '__main__.CheckedTypeA'
    var_1 = '__main__.CheckedTypeB'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = [var_3]
    var_5 = 2
    var_6 = [var_5]

def test_case_0():
    var_0 = '__main__.CheckedType'
    var_1 = 1
    var_2 = [var_1]
    var_3 = [var_2]
    var_4 = 5



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_no_type_specified. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_iterable. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_check_types_with_expected_types_and_matching_element. Retrieved 2/14 statements.
# Partially parsed test_check_types_with_expected_types_and_non_matching_element. Retrieved 2/16 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 3/12 statements.
# Partially parsed test_check_types_with_expected_types_and_multiple_matching_elements. Retrieved 2/15 statements.
# Partially parsed test_check_types_with_expected_types_and_mixed_elements_first_non_matching. Retrieved 2/17 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    assert var_1 is True

def test_case_0():
    var_0 = False
    var_1 = True
    assert var_1 is True

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = True
    var_3 = False
    assert var_3 is True

def test_case_0():
    var_0 = True
    var_1 = False
    assert var_1 is True

def test_case_0():
    var_0 = False
    var_1 = True
    assert var_1 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_returns_set_of_serialized_values. Retrieved 8/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'serialized_1'
    var_5 = 'serialized_2'
    var_6 = 'serialized_3'
    var_7 = {var_4, var_5, var_6}



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 4/11 statements.
# Partially parsed test_constructor_type_check_passes. Retrieved 4/9 statements.
# Partially parsed test_constructor_type_check_fails. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_passes. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_fails. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_no_type_or_invariant. Retrieved 4/8 statements.
# Partially parsed test_constructor_returns_same_type. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 4
    var_1 = 5
    var_2 = 6
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 7
    var_1 = 8
    var_2 = 9
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 'invalid'
    var_2 = 12
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.5
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_pass. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_key_type.
# Failed to parse test_constructor_with_checked_value_type.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 7/14 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 1
    var_8 = -1.5
    var_9 = {var_7: var_8}
    var_10 = bool(False)
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Partially parsed test_maybe_parse_user_type_with_mixed_iterable. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'list'
    var_1 = 'dict'
    var_2 = (var_0, var_1)
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('list', 'dict'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_store_invariants_adds_destination_name. Retrieved 8/17 statements.
# Partially parsed test_store_invariants_inherits_from_bases. Retrieved 3/17 statements.
# Partially parsed test_store_invariants_wraps_functions. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_merges_results. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_handles_multiple_inheritance. Retrieved 3/19 statements.
# Partially parsed test_store_invariants_skips_duplicate_classes. Retrieved 2/16 statements.
# Partially parsed test_store_invariants_preserves_existing_destination. Retrieved 4/15 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'inv1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 'inv2'
    var_6 = var_0[var_1]
    var_7 = len(var_6)
    assert var_7 == 1

def test_case_0():
    var_0 = 'invariants'
    var_1 = 'base_inv'
    var_2 = 'derived_inv'

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = 'not callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'invariants'
    var_1 = 'inv_a'
    var_2 = 'inv_b'

def test_case_0():
    var_0 = 'invariants'
    var_1 = 'inv'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'nonexistent'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = var_0[var_2]
    var_6 = len(var_5)
    assert var_6 == 0

def test_case_0():
    var_0 = 'inv1'
    var_1 = 'inv2'
    var_2 = ()
    var_3 = 'invariants'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 6/9 statements.
# Partially parsed test_check_types_with_no_expected_types. Retrieved 5/7 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_mixed_type_objects_and_strings. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_non_iterable. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = False
    var_5 = True
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 'string'
    var_3 = 3.14
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.float'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2.5
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 'builtins.float'
    var_1 = 1
    var_2 = 2.5
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 42
    var_1 = False
    var_2 = True
    var_3 = bool(var_2)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_key_type.
# Partially parsed test_constructor_with_initial_as_same_class_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_undefined_size_constant. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_non_dict_initial. Retrieved 7/12 statements.
# Partially parsed test_constructor_with_checked_type_create_method. Retrieved 3/14 statements.
# Partially parsed test_constructor_with_checked_type_create_method_and_existing_type. Retrieved 3/13 statements.
# Partially parsed test_constructor_with_ignore_extra_in_create. Retrieved 4/9 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.
# Partially parsed test_constructor_serialize. Retrieved 4/9 statements.
# Partially parsed test_constructor_pickle_support. Retrieved 3/11 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 2.25
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 5.0
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda format, k, v: (str(k), str(v))
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_respects_key_type. Retrieved 3/9 statements.
# Partially parsed test_constructor_respects_value_type. Retrieved 3/9 statements.
# Partially parsed test_constructor_with_invariant. Retrieved 3/9 statements.
# Partially parsed test_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/10 statements.
# Partially parsed test_constructor_with_multiple_invariant_violations. Retrieved 4/9 statements.
# Partially parsed test_constructor_type_check_failure_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_failure_value. Retrieved 3/8 statements.
# Failed to parse test_constructor_with_checked_key_type.
# Failed to parse test_constructor_with_checked_value_type.
# Partially parsed test_constructor_with_checked_type_create. Retrieved 3/20 statements.
# Failed to parse test_constructor_with_checked_type_create_not_needed.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/9 statements.
# Partially parsed test_constructor_with_undefined_size. Retrieved 3/9 statements.
# Partially parsed test_constructor_repr. Retrieved 4/9 statements.
# Partially parsed test_constructor_str. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_self_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_checked_pmap_subclass_instance. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.0
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invalid mapping'

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 0.0
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Key negative'
    var_8 = 'Value non-positive'

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'not_a_float'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap({1: 1.5})'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap({1: 1.5})'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_pass. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_key_type. Retrieved 2/15 statements.
# Partially parsed test_constructor_with_checked_value_type. Retrieved 2/15 statements.
# Partially parsed test_constructor_preserves_subclass_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.


def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 5
    var_1 = 0

def test_case_0():
    var_0 = 5
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/10 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/9 statements.
# Partially parsed test_constructor_multiple_invariants. Retrieved 10/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key not positive')
    var_1 = lambda k, v: (v > 0, 'Value not positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 1
    var_8 = -1.5
    var_9 = {var_7: var_8}
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 1
    var_12 = 1.5
    var_13 = {var_11: var_12}



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 9/10 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_mixed_data. Retrieved 11/12 statements.


def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = (var_0, var_3)
    var_5 = []
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = []
    var_7 = (var_0, var_6)
    var_8 = [var_2, var_5, var_7]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = (var_0, var_1)
    var_3 = 'error2'
    var_4 = (var_0, var_3)
    var_5 = 'error3'
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 42
    var_5 = (var_3, var_4)
    var_6 = 'test'
    var_7 = (var_3, var_6)
    var_8 = []
    var_9 = (var_0, var_8)
    var_10 = [var_2, var_5, var_7, var_9]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_tuple_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_invariant_success. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_no_type_or_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_returns_same_type. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

import pyrsistent._pvector as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = module_0.python_pvector(var_3)

def test_case_0():
    var_0 = 1
    var_1 = 'two'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 2.5
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'two'
    var_2 = 3.0
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #2
#--------------------------

# Failed to parse test___new___creates_checked_pset_from_empty_initial.
# Partially parsed test___new___creates_checked_pset_from_iterable_initial. Retrieved 4/9 statements.
# Partially parsed test___new___creates_checked_pset_from_pmap. Retrieved 6/11 statements.
# Partially parsed test___new___enforces_type_check_on_iterable_initial. Retrieved 3/7 statements.
# Partially parsed test___new___enforces_invariant_on_iterable_initial. Retrieved 4/8 statements.
# Partially parsed test___new___handles_duplicates_in_iterable. Retrieved 4/8 statements.
# Partially parsed test___new___returns_empty_checked_pset_for_empty_iterable. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 1
    var_7 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda x: (x > 0, 'Must be positive')
    var_1 = 1
    var_2 = -1
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = []



# Parsed testcases at query #3
#--------------------------

# Failed to parse test__checked_type_create_with_same_class.
# Partially parsed test__checked_type_create_with_checked_type_subclass. Retrieved 9/20 statements.
# Partially parsed test__checked_type_create_without_checked_type. Retrieved 5/10 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 10/21 statements.
# Partially parsed test__checked_type_create_with_matching_type. Retrieved 6/18 statements.


def test_case_0():
    var_0 = '__main__.MockSubType'
    var_1 = [var_0]
    var_2 = '__main__'
    var_3 = 'MockSubType'
    var_4 = [var_3]
    var_5 = __import__(var_2, fromlist=var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = '__main__.MockSubType'
    var_1 = [var_0]
    var_2 = '__main__'
    var_3 = 'MockSubType'
    var_4 = [var_3]
    var_5 = __import__(var_2, fromlist=var_4)
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = True

def test_case_0():
    var_0 = '__main__.MockSubType'
    var_1 = '__main__.OtherType'
    var_2 = [var_0, var_1]
    var_3 = '__main__'
    var_4 = 'MockSubType'
    var_5 = 'OtherType'
    var_6 = [var_4, var_5]
    var_7 = __import__(var_3, fromlist=var_6)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_CheckedType_constructor. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = '__slots__'
    var_2 = 'create'
    var_3 = 'serialize'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 11/20 statements.
# Partially parsed test_serialize_without_format. Retrieved 8/17 statements.
# Partially parsed test_serialize_empty_map. Retrieved 3/12 statements.
# Partially parsed test_serialize_uses_defined_serializer. Retrieved 8/21 statements.
# Partially parsed test_serialize_preserves_order_from_items. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'fmt'
    var_6 = '1_fmt'
    var_7 = '2_fmt'
    var_8 = 'a_fmt'
    var_9 = 'b_fmt'
    var_10 = {var_6: var_8, var_7: var_9}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '10:None'
    var_6 = '20:None'
    var_7 = {var_0: var_5, var_1: var_6}

def test_case_0():
    var_0 = {}
    var_1 = 'any'
    var_2 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'fixed_key'
    var_6 = 'fixed_value'
    var_7 = {var_5: var_6, var_5: var_6}

def test_case_0():
    var_0 = 3
    var_1 = 1
    var_2 = 2
    var_3 = 'c'
    var_4 = 'a'
    var_5 = 'b'
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test__restore_pickle. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_data'
    var_1 = set()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_invariant_errors_empty_invariants. Retrieved 1/3 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'ok1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'ok'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['error1'])
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'err2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 'err3'
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0._invariant_errors(var_0, var_11)
    var_13 = bool(var_12 == ['err1', 'err2', 'err3'])
    assert var_13 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = True
    var_2 = 'pass1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'fail1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'pass2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = 'fail2'
    var_13 = (var_5, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_4, var_8, var_11, var_14]
    var_16 = module_0._invariant_errors(var_0, var_15)
    var_17 = bool(var_16 == ['fail1', 'fail2'])
    assert var_17 is True

def test_case_0():
    var_0 = []
    var_1 = []

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'all good'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'fine'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_store_types_single_dict. Retrieved 3/5 statements.
# Partially parsed test_store_types_multiple_dicts. Retrieved 4/9 statements.
# Partially parsed test_store_types_iterable_source. Retrieved 3/6 statements.
# Partially parsed test_store_types_preserved_iterable. Retrieved 3/5 statements.
# Partially parsed test_store_types_mixed_sources. Retrieved 9/14 statements.
# Partially parsed test_store_types_nested_iterable. Retrieved 3/7 statements.
# Partially parsed test_store_types_overwrites_destination. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'destination'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'destination'
    var_3 = 'source'
    var_4 = module_0._store_types(var_0, var_1, var_2, var_3)
    var_5 = var_0['destination']
    var_6 = bool(var_0['destination'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 'MyType'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['destination']
    var_7 = bool(var_2['destination'] == ('MyType',))
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Custom'
    var_4 = {var_0: var_3}
    var_5 = [var_1, var_2, var_4]
    var_6 = 'Base2'
    var_7 = ()
    var_8 = {}
    var_9 = [var_6, var_7, var_8]
    var_10 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'destination'
    var_2 = 'old'
    var_3 = []

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'source'
    var_1 = 123
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = 'source'
    var_6 = module_0._store_types(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_initial_is_pmap. Retrieved 6/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 1
    var_7 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_store_invariants_single_class_no_inheritance. Retrieved 8/13 statements.
# Partially parsed test_store_invariants_multiple_inheritance. Retrieved 7/17 statements.
# Partially parsed test_store_invariants_inheritance_chain. Retrieved 7/17 statements.
# Partially parsed test_store_invariants_diamond_inheritance. Retrieved 7/19 statements.
# Partially parsed test_store_invariants_with_local_definition. Retrieved 12/17 statements.
# Partially parsed test_store_invariants_no_invariants. Retrieved 3/7 statements.
# Partially parsed test_store_invariants_non_callable_raises_typeerror. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_wrap_invariant_merges_results. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_wrap_invariant_single_bool_result. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_custom_destination_and_source_names. Retrieved 12/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_0[var_1][var_5]
    var_7 = var_6()
    var_8 = bool(var_7 == (True, ()))
    assert var_8 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]
    var_6 = [inv() for inv in var_5]
    var_7 = True
    var_8 = ()
    var_9 = (var_7, var_8)
    var_10 = bool((True, ()) in var_6)
    assert var_10 is True
    var_11 = False
    var_12 = 'error'
    var_13 = (var_12,)
    var_14 = (var_11, var_13)
    var_15 = bool((False, ('error',)) in var_6)
    assert var_15 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]
    var_6 = [inv() for inv in var_5]
    var_7 = True
    var_8 = ()
    var_9 = (var_7, var_8)
    var_10 = bool((True, ()) in var_6)
    assert var_10 is True
    var_11 = False
    var_12 = 'parent error'
    var_13 = (var_12,)
    var_14 = (var_11, var_13)
    var_15 = bool((False, ('parent error',)) in var_6)
    assert var_15 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2
    var_5 = var_0[var_1]
    var_6 = [inv() for inv in var_5]
    var_7 = True
    var_8 = ()
    var_9 = (var_7, var_8)
    var_10 = bool((True, ()) in var_6)
    assert var_10 is True
    var_11 = False
    var_12 = 'C error'
    var_13 = (var_12,)
    var_14 = (var_11, var_13)
    var_15 = bool((False, ('C error',)) in var_6)
    assert var_15 is True

def test_case_0():
    var_0 = 'invariant'
    var_1 = False
    var_2 = 'local error'
    var_3 = (var_2,)
    var_4 = (var_1, var_3)
    var_5 = lambda self: var_4
    var_6 = {var_0: var_5}
    var_7 = 'invariants'
    var_8 = var_6[var_7]
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_6[var_7]
    var_11 = [inv() for inv in var_10]
    var_12 = True
    var_13 = ()
    var_14 = (var_12, var_13)
    var_15 = bool((True, ()) in var_11)
    assert var_15 is True
    var_16 = False
    var_17 = 'local error'
    var_18 = (var_17,)
    var_19 = (var_16, var_18)
    var_20 = bool((False, ('local error',)) in var_11)
    assert var_20 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0['invariants']
    var_4 = bool(var_0['invariants'] == ())
    assert var_4 is True

def test_case_0():
    var_0 = 'not a function'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0

def test_case_0():
    var_0 = 'custom_invariant'
    var_1 = False
    var_2 = 'local'
    var_3 = (var_2,)
    var_4 = (var_1, var_3)
    var_5 = lambda self: var_4
    var_6 = {var_0: var_5}
    var_7 = 'dest'
    var_8 = var_6[var_7]
    var_9 = len(var_8)
    assert var_9 == 2
    var_10 = var_6[var_7]
    var_11 = [inv() for inv in var_10]
    var_12 = True
    var_13 = ()
    var_14 = (var_12, var_13)
    var_15 = bool((True, ()) in var_11)
    assert var_15 is True
    var_16 = False
    var_17 = 'local'
    var_18 = (var_17,)
    var_19 = (var_16, var_18)
    var_20 = bool((False, ('local',)) in var_11)
    assert var_20 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Failed to parse test_constructor_with_no_arguments.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_checked_type_creation. Retrieved 3/19 statements.
# Partially parsed test_constructor_preserves_original_if_already_instance. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = 'raw_key'
    var_1 = 'raw_value'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #12
#--------------------------

# Failed to parse test___new___creates_empty_checkedpmap.
# Partially parsed test___new___creates_checkedpmap_from_dict. Retrieved 5/10 statements.
# Partially parsed test___new___creates_checkedpmap_from_iterable_of_pairs. Retrieved 7/13 statements.
# Partially parsed test___new___with_size_parameter. Retrieved 4/9 statements.
# Partially parsed test___new___enforces_key_type. Retrieved 3/8 statements.
# Partially parsed test___new___enforces_value_type. Retrieved 3/8 statements.
# Partially parsed test___new___enforces_invariant. Retrieved 4/9 statements.
# Partially parsed test___new___passes_invariant. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 2.25
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serialize_with_default_format. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 9/13 statements.
# Partially parsed test_serialize_with_format_argument. Retrieved 10/14 statements.
# Partially parsed test_serialize_empty_set. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_complex_serializer. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = lambda format, v: str(v)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = {var_5, var_6, var_7}

def test_case_0():
    var_0 = lambda format, v: f'{format}{v}'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 'num_'
    var_6 = 'num_1'
    var_7 = 'num_2'
    var_8 = 'num_3'
    var_9 = {var_6, var_7, var_8}

def test_case_0():
    var_0 = []
    var_1 = set()

def test_case_0():
    var_0 = lambda format, v: {'value': v}
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'value'
    var_5 = {var_4: var_1}
    var_6 = {var_4: var_2}
    var_7 = [var_5, var_6]
    var_8 = len(var_7)



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_store_invariants_inherits_and_stores_callables. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'invariant'
    var_1 = {}
    var_2 = 'dest'
    var_3 = 'dest'
    var_4 = bool('dest' in var_1)
    assert var_4 is True
    var_5 = var_1[var_2]
    var_6 = var_1[var_2]
    var_7 = len(var_6)
    assert var_7 == 2
    var_8 = var_1[var_2]

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'invariant'
    var_1 = True
    var_2 = lambda : var_1
    var_3 = {var_0: var_2}
    var_4 = 'not_callable'
    var_5 = {var_0: var_4}
    var_6 = {}
    var_7 = (var_3, var_5)
    var_8 = 'dest'
    var_9 = 'invariant'
    var_10 = module_0.store_invariants(var_6, var_7, var_8, var_9)
    var_11 = bool(False)
    assert var_11 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'invariant'
    var_2 = True
    var_3 = lambda : var_2
    var_4 = {var_1: var_3}
    var_5 = {}
    var_6 = (var_0, var_4)
    var_7 = 'dest'
    var_8 = module_0.store_invariants(var_5, var_6, var_7, var_1)
    var_9 = 'dest'
    var_10 = bool('dest' in var_5)
    assert var_10 is True
    var_11 = var_5[var_7]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 0
    var_14 = var_5[var_7][var_13]
    var_15 = callable(var_14)
    var_16 = bool(var_15)
    assert var_16 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'dct'
    var_1 = lambda : var_0
    var_2 = 'base'
    var_3 = lambda : var_2
    var_4 = 'invariant'
    var_5 = {var_4: var_1}
    var_6 = {var_4: var_3}
    var_7 = (var_6,)
    var_8 = 'dest'
    var_9 = module_0.store_invariants(var_5, var_7, var_8, var_4)
    var_10 = 0
    var_11 = var_5[var_8][var_10]
    var_12 = var_11()
    assert var_12 == 'dct'
    var_13 = var_5[var_8]
    var_14 = len(var_13)
    assert var_14 == 2

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 'invariant'
    var_3 = {var_2: var_1}
    var_4 = ()
    var_5 = 'dest'
    var_6 = module_0.store_invariants(var_3, var_4, var_5, var_2)
    var_7 = var_3[var_5][var_0]
    var_8 = 5
    var_9 = var_7(var_8)
    assert var_9 is True
    var_10 = -1
    var_11 = var_7(var_10)
    assert var_11 is False



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__check_types_with_matching_types. Retrieved 5/9 statements.
# Partially parsed test__check_types_with_non_matching_type. Retrieved 3/8 statements.
# Partially parsed test__check_types_with_no_expected_types. Retrieved 5/8 statements.
# Partially parsed test__check_types_with_empty_iterable. Retrieved 1/5 statements.
# Partially parsed test__check_types_with_string_type_names. Retrieved 6/9 statements.
# Partially parsed test__check_types_with_mixed_type_and_string. Retrieved 4/8 statements.
# Partially parsed test__check_types_with_custom_exception_type. Retrieved 3/10 statements.
# Partially parsed test__check_types_with_none_in_iterable. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'hello'
    var_3 = 2.5
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 'hello'
    var_5 = [var_3, var_4]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 1
    var_2 = 'hello'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_0, var_3]



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_tuple_initial. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_set_initial. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 6/10 statements.
# Partially parsed test_constructor_type_check_pass. Retrieved 4/8 statements.
# Partially parsed test_constructor_type_check_fail. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_pass. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_fail. Retrieved 5/9 statements.
# Partially parsed test_constructor_duplicate_elements. Retrieved 4/8 statements.
# Partially parsed test_constructor_no_type_or_invariant. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = {var_0, var_1, var_2}
    var_4 = 1
    var_5 = 2
    var_6 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = True
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 1
    var_7 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2.5
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2.5
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.5
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 'a'
    var_6 = 3.5



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_checked_type_create_with_checked_type_and_mismatched_data. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'data1'
    var_1 = 'data2'
    var_2 = [var_0, var_1]
    var_3 = False



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_invariant_exception_constructor_with_callable_error_codes.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.InvariantException(**var_0)
    var_2 = var_1.invariant_errors
    var_3 = bool(var_1.invariant_errors == ())
    assert var_3 is True
    var_4 = var_1.missing_fields
    var_5 = bool(var_1.missing_fields == ())
    assert var_5 is True
    var_6 = str(var_1)
    assert var_6 == ', invariant_errors=[], missing_fields=[]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.InvariantException(var_2, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ('error1', 'error2'))
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ())
    assert var_8 is True
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[error1, error2], missing_fields=[]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.InvariantException(missing_fields=var_2, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ())
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ('field1', 'field2'))
    assert var_8 is True
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[], missing_fields=[field1, field2]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'err1'
    var_1 = 'err2'
    var_2 = (var_0, var_1)
    var_3 = 'missing1'
    var_4 = (var_3,)
    var_5 = {}
    var_6 = module_0.InvariantException(var_2, var_4, **var_5)
    var_7 = var_6.invariant_errors
    var_8 = bool(var_6.invariant_errors == ('err1', 'err2'))
    assert var_8 is True
    var_9 = var_6.missing_fields
    var_10 = bool(var_6.missing_fields == ('missing1',))
    assert var_10 is True
    var_11 = str(var_6)
    assert var_11 == ', invariant_errors=[err1, err2], missing_fields=[missing1]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'extra_arg'
    var_1 = 'extra_kwarg'
    var_2 = 'another'
    var_3 = {var_2: var_1}
    var_4 = module_0.InvariantException(var_0, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ())
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ())
    assert var_8 is True
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[], missing_fields=[]'



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_constructor_with_callable_error_codes.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.InvariantException(**var_0)
    var_2 = var_1.invariant_errors
    var_3 = bool(var_1.invariant_errors == ())
    assert var_3 is True
    var_4 = var_1.missing_fields
    var_5 = bool(var_1.missing_fields == ())
    assert var_5 is True
    var_6 = str(var_1)
    assert var_6 == ', invariant_errors=[], missing_fields=[]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.InvariantException(var_2, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ('error1', 'error2'))
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ())
    assert var_8 is True
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[error1, error2], missing_fields=[]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = module_0.InvariantException(missing_fields=var_2, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ())
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ('field1', 'field2'))
    assert var_8 is True
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[], missing_fields=[field1, field2]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'err1'
    var_1 = 'err2'
    var_2 = (var_0, var_1)
    var_3 = 'missing1'
    var_4 = (var_3,)
    var_5 = {}
    var_6 = module_0.InvariantException(var_2, var_4, **var_5)
    var_7 = var_6.invariant_errors
    var_8 = bool(var_6.invariant_errors == ('err1', 'err2'))
    assert var_8 is True
    var_9 = var_6.missing_fields
    var_10 = bool(var_6.missing_fields == ('missing1',))
    assert var_10 is True
    var_11 = str(var_6)
    assert var_11 == ', invariant_errors=[err1, err2], missing_fields=[missing1]'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'extra_arg'
    var_1 = 'extra_kwarg'
    var_2 = 'another'
    var_3 = {var_2: var_1}
    var_4 = module_0.InvariantException(var_0, **var_3)
    var_5 = var_4.invariant_errors
    var_6 = bool(var_4.invariant_errors == ())
    assert var_6 is True
    var_7 = var_4.missing_fields
    var_8 = bool(var_4.missing_fields == ())
    assert var_8 is True
    var_9 = str(var_4)
    assert var_9 == ', invariant_errors=[], missing_fields=[]'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serialize_with_default_format. Retrieved 9/13 statements.
# Partially parsed test_serialize_with_custom_format. Retrieved 8/12 statements.
# Partially parsed test_serialize_empty_set. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_complex_serializer. Retrieved 8/12 statements.


def test_case_0():
    var_0 = lambda format, v: str(v)
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '1'
    var_6 = '2'
    var_7 = '3'
    var_8 = {var_5, var_6, var_7}

def test_case_0():
    var_0 = lambda format, v: f'{v}:{format}'
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'fmt'
    var_5 = '1:fmt'
    var_6 = '2:fmt'
    var_7 = {var_5, var_6}

def test_case_0():
    var_0 = lambda format, v: str(v)
    var_1 = set()

def test_case_0():
    var_0 = lambda format, v: v.upper() if format == 'upper' else v
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'upper'
    var_5 = 'A'
    var_6 = 'B'
    var_7 = {var_5, var_6}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_duplicates_in_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 7/12 statements.
# Partially parsed test_constructor_type_violation. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_violation. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_no_type_or_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_iterable_initial. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_2, var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = 1
    var_8 = 2

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'a'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = 1
    var_6 = 2
    var_7 = 3



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_types_with_expected_types_and_matching_element. Retrieved 1/12 statements.
# Partially parsed test_check_types_with_expected_types_and_non_matching_element. Retrieved 1/15 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 2/10 statements.
# Partially parsed test_check_types_with_multiple_expected_types_and_matching_element. Retrieved 1/14 statements.
# Partially parsed test_check_types_with_multiple_expected_types_and_non_matching_element. Retrieved 1/17 statements.
# Partially parsed test_check_types_with_none_expected_types. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'MockSourceClass'

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 'MockSourceClass'

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = []
    var_2 = None



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_invariant_errors_passes_elem_to_each_invariant. Retrieved 1/7 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'test_element'
    var_1 = True
    var_2 = 'ok1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ok2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == [])
    assert var_10 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'valid'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'error2'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)
    var_14 = bool(var_13 == ['error1', 'error2'])
    assert var_14 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = []

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = 'ignore_me'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'include_me'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0._invariant_errors(var_0, var_9)
    var_11 = bool(var_10 == ['include_me'])
    assert var_11 is True



# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------

# Partially parsed test_merge_invariant_results_all_true. Retrieved 8/9 statements.
# Partially parsed test_merge_invariant_results_one_false. Retrieved 10/11 statements.
# Partially parsed test_merge_invariant_results_multiple_false. Retrieved 11/12 statements.
# Partially parsed test_merge_invariant_results_empty. Retrieved 1/2 statements.
# Partially parsed test_merge_invariant_results_mixed_data. Retrieved 12/13 statements.
# Partially parsed test_merge_invariant_results_only_false_single. Retrieved 5/6 statements.


def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = []
    var_4 = (var_0, var_3)
    var_5 = []
    var_6 = (var_0, var_5)
    var_7 = [var_2, var_4, var_6]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error1'
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = []
    var_8 = (var_0, var_7)
    var_9 = [var_2, var_6, var_8]

def test_case_0():
    var_0 = False
    var_1 = 'error1'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = 'error2'
    var_5 = [var_4]
    var_6 = (var_0, var_5)
    var_7 = 'error3'
    var_8 = [var_7]
    var_9 = (var_0, var_8)
    var_10 = [var_3, var_6, var_9]

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = 'error_a'
    var_5 = [var_4]
    var_6 = (var_3, var_5)
    var_7 = 'error_b'
    var_8 = 'error_c'
    var_9 = [var_7, var_8]
    var_10 = (var_3, var_9)
    var_11 = [var_2, var_6, var_10]

def test_case_0():
    var_0 = False
    var_1 = 'only_error'
    var_2 = [var_1]
    var_3 = (var_0, var_2)
    var_4 = [var_3]



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_returns_dict_from_serializer. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_format'
    var_6 = (var_0, var_2, var_5)
    var_7 = (var_1, var_3, var_5)
    var_8 = {var_0: var_6, var_1: var_7}



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_key_type. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_value_type. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 1/7 statements.
# Partially parsed test_constructor_with_initial_and_size_parameter. Retrieved 4/9 statements.
# Partially parsed test_constructor_returns_same_instance_if_already_checked_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_none_initial. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = None



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_returns_dict_from_serializer. Retrieved 9/16 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'test_format'
    var_6 = (var_0, var_2, var_5)
    var_7 = (var_1, var_3, var_5)
    var_8 = {var_0: var_6, var_1: var_7}



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Partially parsed test_maybe_parse_user_type_with_mixed_iterable. Retrieved 1/4 statements.
# Partially parsed test_maybe_parse_user_type_raises_type_error_for_invalid_iterable_element. Retrieved 1/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True

def test_case_0():
    var_0 = 'str'

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 123
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 2/5 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_set_initial. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 6/9 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 3/7 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 3/7 statements.
# Partially parsed test_constructor_duplicate_elements. Retrieved 4/7 statements.
# Partially parsed test_constructor_with_no_type_or_invariant. Retrieved 4/7 statements.
# Partially parsed test_constructor_repr_output. Retrieved 4/8 statements.
# Partially parsed test_constructor_str_output. Retrieved 4/8 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = set()

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 4
    var_2 = 5
    var_3 = {var_1, var_2}

import pyrsistent._pset as module_0

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 6
    var_2 = 7
    var_3 = [var_1, var_2]
    var_4 = module_0.pset(var_3)
    var_5 = var_4._map

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 'a'
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = -1
    var_2 = [var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_1, var_2, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 3
    var_2 = 4
    var_3 = [var_1, var_2]



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_wrap_invariant_single_bool_true.
# Failed to parse test_wrap_invariant_single_bool_false.
# Failed to parse test_wrap_invariant_multiple_results_all_true.
# Failed to parse test_wrap_invariant_multiple_results_one_false.
# Failed to parse test_wrap_invariant_multiple_results_multiple_false.
# Partially parsed test_wrap_invariant_with_args. Retrieved 2/6 statements.
# Partially parsed test_wrap_invariant_with_kwargs. Retrieved 1/5 statements.
# Failed to parse test_wrap_invariant_empty_result_list.
# Failed to parse test_wrap_invariant_single_tuple_in_list.


def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 5



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/10 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_check_passes. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/9 statements.
# Partially parsed test_constructor_repr. Retrieved 4/9 statements.
# Partially parsed test_constructor_str. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'string'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap({1: 1.5})'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap({1: 1.5})'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_no_expected_types. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_string_type_names. Retrieved 7/9 statements.
# Partially parsed test_check_types_with_mixed_type_and_string. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 4/10 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_single_invalid_element. Retrieved 3/7 statements.
# Partially parsed test_check_types_with_all_invalid_elements. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 2
    var_3 = 'world'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 'any'
    var_3 = 3.14
    var_4 = None
    var_5 = [var_1, var_2, var_3, var_4]

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 'valid'
    var_5 = 2
    var_6 = [var_3, var_4, var_5]

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 1
    var_2 = 'mixed'
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'error'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 3.14
    var_1 = 'not_float'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'false'
    var_1 = 'true'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #34
#--------------------------

# Failed to parse test_predicate_at_line_2_evaluates_to_false.




# Parsed testcases at query #35
#--------------------------

# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises_error. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_empty_dict. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/10 statements.
# Partially parsed test_constructor_with_invariant_list_violation. Retrieved 4/9 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_initial_as_same_class_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_checked_type_value. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_mixed_valid_types. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_invalid_mixed_type. Retrieved 2/7 statements.
# Partially parsed test_constructor_with_no_type_specification. Retrieved 5/8 statements.
# Partially parsed test_constructor_with_inheritance. Retrieved 3/10 statements.
# Partially parsed test_constructor_with_create_method. Retrieved 4/14 statements.
# Partially parsed test_constructor_with_checked_key_type_create. Retrieved 1/12 statements.
# Partially parsed test_constructor_with_serialize_method. Retrieved 4/9 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = {var_0: var_1}
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = 2.5
    var_3 = {var_1: var_2}
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

def test_case_0():
    var_0 = {}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 1.5
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 1.5
    var_3 = 'test'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1.5
    var_1 = {var_0: var_0}
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 0

def test_case_0():
    var_0 = 'any'

def test_case_0():
    var_0 = lambda format, k, v: (k, v)
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



