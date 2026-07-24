####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
# Partially parsed test_constructor_repr. Retrieved 6/12 statements.


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
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'IntToFloatMap({1: 1.5, 2: 2.25})'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test__store_types_with_single_dict. Retrieved 3/5 statements.
# Partially parsed test__store_types_with_multiple_dicts. Retrieved 7/13 statements.
# Partially parsed test__store_types_with_iterable_source. Retrieved 3/6 statements.
# Partially parsed test__store_types_with_preserved_iterable_type. Retrieved 3/5 statements.
# Partially parsed test__store_types_with_no_source_in_dict_or_bases. Retrieved 6/9 statements.
# Partially parsed test__store_types_with_source_in_base_only. Retrieved 5/9 statements.
# Partially parsed test__store_types_with_nested_iterable_source. Retrieved 3/7 statements.
# Partially parsed test__store_types_overwrites_destination_key. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Base2'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

import pyrsistent._checked_types as module_0


def test_case_0():
    var_0 = 'source'
    var_1 = 'CustomType'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['destination']
    var_7 = bool(var_2['destination'] == ('CustomType',))
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = {}
    var_4 = [var_1, var_2, var_3]
    var_5 = 'destination'
    var_6 = 'source'
    var_7 = 'destination'
    var_8 = bool('destination' not in var_0)
    assert var_8 is True

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'source'
    var_4 = 'destination'
    var_5 = var_0['destination']

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'


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

def test_case_0():
    var_0 = 'source'
    var_1 = 'destination'
    var_2 = 'old_value'
    var_3 = []



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_invariant_errors_elem_passed_correctly. Retrieved 4/9 statements.



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


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'err2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0._invariant_errors(var_0, var_8)
    var_10 = bool(var_9 == ['err1', 'err2'])
    assert var_10 is True


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'ok'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = 'err'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = True
    var_12 = 'fine'
    var_13 = (var_11, var_12)
    var_14 = lambda x: var_13
    var_15 = [var_6, var_10, var_14]
    var_16 = module_0._invariant_errors(var_2, var_15)
    var_17 = bool(var_16 == ['err'])
    assert var_17 is True


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(var_0 == [var_3])
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_store_invariants_adds_destination_with_wrapped_invariants. Retrieved 22/26 statements.
# Partially parsed test_store_invariants_inherits_from_multiple_bases. Retrieved 24/28 statements.
# Partially parsed test_store_invariants_handles_duplicate_inheritance. Retrieved 16/19 statements.
# Partially parsed test_store_invariants_wraps_invariant_returning_list. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_wraps_invariant_returning_bool_tuple. Retrieved 5/11 statements.


def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = lambda self: var_2
    var_4 = False
    var_5 = 'error'
    var_6 = (var_5,)
    var_7 = (var_4, var_6)
    var_8 = lambda self: var_7
    var_9 = 'Base'
    var_10 = ()
    var_11 = 'invariant'
    var_12 = {var_11: var_3}
    var_13 = [var_9, var_10, var_12]
    var_14 = {var_11: var_8}
    var_15 = 'invariants'
    var_16 = var_14[var_15]
    var_17 = len(var_16)
    assert var_17 == 2
    var_18 = var_16[var_4]
    var_19 = None
    var_20 = var_18(var_19)
    var_21 = bool(var_20 == (True, ()))
    assert var_21 is True
    var_22 = var_16[var_0]
    var_23 = var_22(var_19)
    var_24 = bool(var_23 == (False, ('error',)))
    assert var_24 is True


def test_case_0():
    var_0 = ()
    var_1 = 'invariant'
    var_2 = 'not callable'
    var_3 = {var_1: var_2}
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_3, var_0, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = lambda self: var_2
    var_4 = ()
    var_5 = (var_0, var_4)
    var_6 = lambda self: var_5
    var_7 = 'Base1'
    var_8 = ()
    var_9 = 'invariant'
    var_10 = {var_9: var_3}
    var_11 = [var_7, var_8, var_10]
    var_12 = 'Base2'
    var_13 = ()
    var_14 = {var_9: var_6}
    var_15 = [var_12, var_13, var_14]
    var_16 = {}
    var_17 = 'invariants'
    var_18 = var_16[var_17]
    var_19 = len(var_18)
    assert var_19 == 2
    var_20 = 0
    var_21 = var_18[var_20]
    var_22 = None
    var_23 = var_21(var_22)
    var_24 = bool(var_23 == (True, ()))
    assert var_24 is True
    var_25 = var_18[var_0]
    var_26 = var_25(var_22)
    var_27 = bool(var_26 == (True, ()))
    assert var_27 is True

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = lambda self: var_2
    var_4 = 'Base'
    var_5 = ()
    var_6 = 'invariant'
    var_7 = {var_6: var_3}
    var_8 = [var_4, var_5, var_7]
    var_9 = {}
    var_10 = 'invariants'
    var_11 = var_9[var_10]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = 0
    var_14 = var_11[var_13]
    var_15 = None
    var_16 = var_14(var_15)
    var_17 = bool(var_16 == (True, ()))
    assert var_17 is True


def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_1, var_0, var_2, var_3)
    var_5 = var_1[var_2]
    var_6 = bool(var_5 == ())
    assert var_6 is True

def test_case_0():
    var_0 = ()
    var_1 = 'invariant'
    var_2 = 'invariants'
    var_3 = 0
    var_4 = None

def test_case_0():
    var_0 = ()
    var_1 = 'invariant'
    var_2 = 'invariants'
    var_3 = 0
    var_4 = None



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_success. Retrieved 4/9 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_no_type_or_invariant. Retrieved 4/8 statements.
# Partially parsed test_constructor_returns_same_type. Retrieved 4/8 statements.


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
    var_1 = 1.5
    var_2 = 2
    var_3 = 3.0
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
    var_1 = 'a'
    var_2 = 2
    var_3 = 'b'
    var_4 = [var_0, var_1, var_2, var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = None
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_correct_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_wrong_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_wrong_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_key_type. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_value_type. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_types_and_create. Retrieved 3/20 statements.
# Partially parsed test_constructor_with_initial_as_same_checkedpmap_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_repr. Retrieved 4/10 statements.
# Partially parsed test_constructor_str. Retrieved 4/10 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/8 statements.
# Partially parsed test_constructor_serialize. Retrieved 4/9 statements.
# Partially parsed test_constructor_pickle_support. Retrieved 3/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 3.14
    var_3 = {var_0: var_0, var_1: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
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
    var_2 = 2.0
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
    var_3 = 'IntToFloatMap'
    var_4 = '1: 1.5'

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'IntToFloatMap'
    var_4 = '1: 1.5'

def test_case_0():
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = lambda format, k, v: (k, v)
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_invariant_errors_returns_empty_list_for_empty_invariants. Retrieved 1/3 statements.
# Partially parsed test_invariant_errors_passes_elem_to_each_invariant. Retrieved 5/10 statements.


import pyrsistent._checked_types as module_0


def test_case_0():
    var_0 = 'key'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'ok'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_6, var_9]
    var_11 = module_0._invariant_errors(var_2, var_10)
    var_12 = bool(var_11 == [])
    assert var_12 is True


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


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 'err1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = 'ignored'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'err2'
    var_10 = (var_1, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)
    var_14 = bool(var_13 == ['err1', 'err2'])
    assert var_14 is True

def test_case_0():
    var_0 = []
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = bool(var_0 == [var_4, var_4])
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_type.
# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_nested_iterable.
# Partially parsed test_maybe_parse_user_type_mixed_valid_iterable. Retrieved 1/4 statements.



def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 2/6 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 6/10 statements.
# Partially parsed test_constructor_with_set_initial. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 8/12 statements.
# Partially parsed test_constructor_type_violation. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_violation. Retrieved 4/8 statements.
# Partially parsed test_constructor_duplicate_elements. Retrieved 4/8 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_1, var_2, var_3}

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1.5
    var_2 = 2.5
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = True
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = {var_4, var_2}

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'a'
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -1
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_1, var_2]
    var_4 = 1
    var_5 = 2



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_invalid_type. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_no_expected_types. Retrieved 7/9 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/4 statements.
# Partially parsed test_check_types_with_string_type_name. Retrieved 6/8 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 3/9 statements.
# Partially parsed test_check_types_with_mixed_type_names_and_types. Retrieved 5/8 statements.
# Partially parsed test_check_types_with_invalid_type_name. Retrieved 4/7 statements.


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
    var_0 = None
    var_1 = 1
    var_2 = 'hello'
    var_3 = 3.14
    var_4 = 2
    var_5 = [var_1, var_4]
    var_6 = [var_1, var_2, var_3, var_5]

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
    var_0 = 1
    var_1 = 'invalid'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 1
    var_2 = 'hello'
    var_3 = 2
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = 'nonexistent.module.Class'
    var_1 = [var_0]
    var_2 = 1
    var_3 = [var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/10 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants_success. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_key_type. Retrieved 1/11 statements.
# Partially parsed test_constructor_with_checked_value_type. Retrieved 1/11 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.


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
    var_0 = lambda k, v: (int(v) == k, 'Invalid mapping')
    var_1 = 1
    var_2 = {var_1: var_1}

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
    var_0 = lambda k, v: (k > 0, 'Key must be positive')
    var_1 = lambda k, v: (v > 0, 'Value must be positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 1.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1

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


def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'ok'
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


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = bool(var_2 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = []


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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_invariant_errors_passes_elem_to_each_invariant. Retrieved 4/9 statements.



def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = (var_3, var_4)
    var_8 = lambda x: var_7
    var_9 = [var_6, var_8]
    var_10 = module_0._invariant_errors(var_2, var_9)
    var_11 = bool(var_10 == [])
    assert var_11 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = None
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = 'error1'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'error2'
    var_12 = (var_7, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_6, var_10, var_13]
    var_15 = module_0._invariant_errors(var_2, var_14)
    var_16 = bool(var_15 == ['error1', 'error2'])
    assert var_16 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = False
    var_4 = 'error1'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = 'error2'
    var_8 = (var_3, var_7)
    var_9 = lambda x: var_8
    var_10 = [var_6, var_9]
    var_11 = module_0._invariant_errors(var_2, var_10)
    var_12 = bool(var_11 == ['error1', 'error2'])
    assert var_12 is True

def test_case_0():
    var_0 = []
    var_1 = 'id'
    var_2 = 123
    var_3 = {var_1: var_2}
    var_4 = bool(var_0 == [var_3, var_3])
    assert var_4 is True


def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = module_0._invariant_errors(var_2, var_3)
    var_5 = bool(var_4 == [])
    assert var_5 is True


def test_case_0():
    var_0 = 'value'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 'ignored'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = False
    var_8 = 'included'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'ignored2'
    var_12 = (var_3, var_11)
    var_13 = lambda x: var_12
    var_14 = [var_6, var_10, var_13]
    var_15 = module_0._invariant_errors(var_2, var_14)
    var_16 = bool(var_15 == ['included'])
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_constructor_with_valid_initial_dict. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_empty_initial_dict. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises_error. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 4/8 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_key_type. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_checked_value_type. Retrieved 1/9 statements.
# Partially parsed test_constructor_with_initial_as_same_class_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_undefined_size_constant. Retrieved 3/7 statements.
# Partially parsed test_constructor_repr_output. Retrieved 3/8 statements.
# Partially parsed test_constructor_str_output. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = {}

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
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2.0
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 2.0
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Key negative'

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1

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

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 2/7 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_set_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 7/12 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_duplicate_elements. Retrieved 5/10 statements.
# Partially parsed test_constructor_no_type_or_invariant. Retrieved 5/10 statements.


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = []

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = {var_1, var_2, var_3}

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 4
    var_2 = 5
    var_3 = {var_1, var_2}
    var_4 = {var_1, var_2}

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 6
    var_2 = 7
    var_3 = True
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = {var_1, var_2}

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'a'
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -1
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_1, var_2, var_2]
    var_4 = {var_1, var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_0, var_1, var_2}



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_with_invariant_success. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_tuple_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_generator_initial. Retrieved 2/8 statements.
# Partially parsed test_constructor_repr_output. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

import pyrsistent._pvector as module_0


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
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
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)

def test_case_0():
    var_0 = 3
    var_1 = range(var_0)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__restore_pickle. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'test_data'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 3/7 statements.
# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Failed to parse test_constructor_with_checked_type_key.
# Partially parsed test_constructor_with_checked_type_creation. Retrieved 3/19 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/7 statements.
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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_store_invariants_adds_destination_tuple. Retrieved 14/26 statements.
# Partially parsed test_store_invariants_inherits_from_all_bases. Retrieved 5/17 statements.
# Partially parsed test_store_invariants_includes_local_dict. Retrieved 2/14 statements.
# Partially parsed test_store_invariants_raises_typeerror_for_noncallable. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_handles_missing_source_name. Retrieved 5/10 statements.
# Partially parsed test_store_invariants_wraps_invariants. Retrieved 5/11 statements.
# Partially parsed test_store_invariants_does_not_duplicate_inherited. Retrieved 8/15 statements.
# Partially parsed test_store_invariants_with_multiple_inheritance_diamond. Retrieved 8/19 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'my_invariants'
    var_2 = 'inv1'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = {}
    var_10 = 'inv2'
    var_11 = var_9[var_1]
    var_12 = len(var_11)
    assert var_12 == 1
    var_13 = var_11[var_5]
    var_14 = callable(var_13)
    var_15 = bool(var_14)
    assert var_15 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = 'invariant'
    var_1 = 'invariants'

def test_case_0():
    var_0 = 'not a function'
    var_1 = {}
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._checked_types as module_0


def test_case_0():
    var_0 = 'invariant'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'nonexistent'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 0

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0
    var_4 = None

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_types_with_no_expected_types. Retrieved 5/6 statements.
# Partially parsed test_check_types_with_matching_types. Retrieved 4/6 statements.
# Partially parsed test_check_types_with_matching_multiple_types. Retrieved 4/6 statements.
# Partially parsed test_check_types_raises_exception_on_mismatch. Retrieved 4/7 statements.
# Partially parsed test_check_types_uses_custom_exception_type. Retrieved 3/6 statements.
# Partially parsed test_check_types_with_none_in_iterable. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Type list can only be used with (int,), not str'

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = [var_0, var_1]
    var_3 = [var_0]



# Parsed testcases at query #21
#--------------------------

# Failed to parse test__checked_type_create_with_same_class.
# Partially parsed test__checked_type_create_with_checked_type_subclass. Retrieved 10/22 statements.
# Partially parsed test__checked_type_create_with_matching_type_in_list. Retrieved 3/7 statements.
# Partially parsed test__checked_type_create_without_checked_types. Retrieved 4/8 statements.
# Partially parsed test__checked_type_create_ignore_extra. Retrieved 11/23 statements.


def test_case_0():
    var_0 = '__main__.MockSubType'
    var_1 = [var_0]
    var_2 = '__main__'
    var_3 = 'MockSubType'
    var_4 = [var_3]
    var_5 = __import__(var_2, fromlist=var_4)
    var_6 = __import__(var_2)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = '__main__.MockSubType'
    var_1 = [var_0]
    var_2 = '__main__'
    var_3 = 'MockSubType'
    var_4 = [var_3]
    var_5 = __import__(var_2, fromlist=var_4)
    var_6 = __import__(var_2)
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_check_types_with_valid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_mixed_valid_types. Retrieved 4/8 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 3/8 statements.
# Partially parsed test_check_types_with_no_elements. Retrieved 1/5 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 'invalid'
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_checkedpmap_initial. Retrieved 3/10 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_size_and_initial. Retrieved 3/9 statements.
# Partially parsed test_constructor_creates_new_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 10/19 statements.


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
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

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
    var_8 = 1
    var_9 = -1.5
    var_10 = {var_8: var_9}
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True
    var_13 = 1
    var_14 = 1.5
    var_15 = {var_13: var_14}



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_wrap_invariant_single_bool_true.
# Failed to parse test_wrap_invariant_single_bool_false.
# Failed to parse test_wrap_invariant_multiple_results_all_true.
# Failed to parse test_wrap_invariant_multiple_results_one_false.
# Failed to parse test_wrap_invariant_multiple_results_all_false.
# Partially parsed test_wrap_invariant_with_args. Retrieved 2/6 statements.
# Partially parsed test_wrap_invariant_with_kwargs. Retrieved 1/5 statements.
# Partially parsed test_wrap_invariant_multiple_results_with_args. Retrieved 1/5 statements.
# Failed to parse test_wrap_invariant_empty_result_list.


def test_case_0():
    var_0 = 5
    var_1 = 3

def test_case_0():
    var_0 = 0

def test_case_0():
    var_0 = 4



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_store_invariants_adds_destination_name. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_includes_inherited_invariants. Retrieved 5/13 statements.
# Partially parsed test_store_invariants_wraps_invariants. Retrieved 10/4 statements.
# Partially parsed test_store_invariants_merges_multiple_results. Retrieved 4/10 statements.
# Partially parsed test_store_invariants_avoids_duplicate_inheritance. Retrieved 5/16 statements.
# Partially parsed test_store_invariants_uses_correct_source_and_destination_names. Retrieved 5/10 statements.
# Partially parsed test_store_invariants_handles_multiple_inheritance. Retrieved 5/15 statements.
# Partially parsed test_store_invariants_works_with_instance_methods. Retrieved 5/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 'invariants'
    var_4 = bool('invariants' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2


def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_3, var_0, var_4, var_5)
    var_7 = 0
    var_8 = var_3[var_4][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (True, ()))
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = {}
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_3, var_0, var_4, var_5)
    var_7 = 0
    var_8 = var_3[var_4][var_7]
    var_9 = var_8()
    var_10 = bool(var_9 == (True, ()))
    assert var_10 is True

def test_case_0():
    var_0 = 'invariant'
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 0


def test_case_0():
    var_0 = 'invariant'
    var_1 = 'not a function'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = 'invariants'
    var_5 = 'invariant'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True


def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'invariants'
    var_3 = 'invariant'
    var_4 = module_0.store_invariants(var_0, var_1, var_2, var_3)
    var_5 = var_0['invariants']
    var_6 = bool(var_0['invariants'] == ())
    assert var_6 is True

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1

def test_case_0():
    var_0 = {}
    var_1 = 'dest'
    var_2 = 'my_invariant'
    var_3 = 'dest'
    var_4 = bool('dest' in var_0)
    assert var_4 is True
    var_5 = var_0[var_1]
    var_6 = len(var_5)
    assert var_6 == 1

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 2

def test_case_0():
    var_0 = {}
    var_1 = 'invariants'
    var_2 = 'invariant'
    var_3 = 0
    var_4 = var_0[var_1][var_3]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_store_types_single_dict. Retrieved 3/5 statements.
# Partially parsed test_store_types_multiple_dicts. Retrieved 7/13 statements.
# Partially parsed test_store_types_no_source. Retrieved 8/11 statements.
# Partially parsed test_store_types_iterable_source. Retrieved 3/6 statements.
# Partially parsed test_store_types_preserved_iterable. Retrieved 3/5 statements.
# Partially parsed test_store_types_mixed_sources. Retrieved 5/11 statements.
# Partially parsed test_store_types_overwrites_destination. Retrieved 4/6 statements.
# Partially parsed test_store_types_nested_iterable. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Base1'
    var_2 = ()
    var_3 = 'Base2'
    var_4 = ()
    var_5 = {}
    var_6 = [var_3, var_4, var_5]
    var_7 = 'destination'

def test_case_0():
    var_0 = {}
    var_1 = 'Base'
    var_2 = ()
    var_3 = 'other'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = [var_1, var_2, var_5]
    var_7 = 'destination'
    var_8 = 'source'
    var_9 = 'destination'
    var_10 = bool('destination' not in var_0)
    assert var_10 is True


def test_case_0():
    var_0 = 'source'
    var_1 = 'MyClass'
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['destination']
    var_7 = bool(var_2['destination'] == ('MyClass',))
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'Custom'
    var_2 = 'Base'
    var_3 = ()
    var_4 = 'destination'

def test_case_0():
    var_0 = 'source'
    var_1 = 'destination'
    var_2 = 'old'
    var_3 = []


def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'destination'
    var_5 = module_0._store_types(var_2, var_3, var_4, var_0)
    var_6 = var_2['destination']
    var_7 = bool(var_2['destination'] == ())
    assert var_7 is True

def test_case_0():
    var_0 = 'source'
    var_1 = []
    var_2 = 'destination'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_with_valid_key_value_types. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_invalid_key_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type_raises_error. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation_raises_error. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Failed to parse test_constructor_with_no_arguments.
# Partially parsed test_constructor_with_checked_type_key_and_value. Retrieved 5/14 statements.
# Partially parsed test_constructor_with_checked_type_key_and_value_using_create. Retrieved 5/14 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 10/18 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 5/10 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.
# Partially parsed test_constructor_serialize. Retrieved 4/9 statements.
# Partially parsed test_constructor_pickling. Retrieved 3/11 statements.


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
    var_0 = {}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}

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
    var_8 = 1
    var_9 = -1.5
    var_10 = {var_8: var_9}
    var_11 = bool(False)
    assert var_11 is True
    var_12 = bool(True)
    assert var_12 is True
    var_13 = 1
    var_14 = 1.5
    var_15 = {var_13: var_14}

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
    var_0 = lambda format, k, v: (k, v)
    var_1 = 1
    var_2 = 1.5
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_invariant_errors_passes_elem_to_invariants. Retrieved 2/7 statements.



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
    var_10 = []
    var_11 = bool(var_9 == var_10)
    assert var_11 is True


def test_case_0():
    var_0 = 'test'
    var_1 = True
    var_2 = 'ok'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'err1'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'err2'
    var_10 = (var_5, var_9)
    var_11 = lambda x: var_10
    var_12 = [var_4, var_8, var_11]
    var_13 = module_0._invariant_errors(var_0, var_12)
    var_14 = [var_6, var_9]
    var_15 = bool(var_13 == var_14)
    assert var_15 is True


def test_case_0():
    var_0 = None
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
    var_0 = 10
    var_1 = []
    var_2 = module_0._invariant_errors(var_0, var_1)
    var_3 = []
    var_4 = bool(var_2 == var_3)
    assert var_4 is True

def test_case_0():
    var_0 = []
    var_1 = 42
    var_2 = bool(var_0 == [var_1])
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_maybe_parse_user_type_preserved_type.
# Failed to parse test_maybe_parse_user_type_single_type.
# Failed to parse test_maybe_parse_user_type_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_nested_iterable.
# Partially parsed test_maybe_parse_user_type_mixed_iterable. Retrieved 1/3 statements.
# Failed to parse test_maybe_parse_user_type_deeply_nested.



def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True


def test_case_0():
    var_0 = []
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_list_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_python_pvector_initial. Retrieved 4/11 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 5/9 statements.
# Partially parsed test_constructor_invariant_check_success. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_multiple_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_no_type_specified. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

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



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_maybe_parse_user_type_with_preserved_type.
# Failed to parse test_maybe_parse_user_type_with_single_type.
# Failed to parse test_maybe_parse_user_type_with_iterable_of_types.
# Failed to parse test_maybe_parse_user_type_with_nested_iterable.
# Partially parsed test_maybe_parse_user_type_with_mixed_iterable. Retrieved 1/3 statements.



def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True


def test_case_0():
    var_0 = ()
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ())
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True


def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = [var_0, var_1]
    var_3 = module_0.maybe_parse_user_type(var_2)
    var_4 = bool(var_3 == ('int', 'str'))
    assert var_4 is True

def test_case_0():
    var_0 = 'str'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_invariant_errors_empty_invariants. Retrieved 1/3 statements.
# Partially parsed test_invariant_errors_elem_passed_to_invariants. Retrieved 4/9 statements.



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

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = bool(var_0 == [var_3, var_3])
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_types_with_no_expected_types. Retrieved 5/6 statements.
# Partially parsed test_check_types_with_single_matching_type. Retrieved 4/6 statements.
# Partially parsed test_check_types_with_multiple_matching_types. Retrieved 4/6 statements.
# Partially parsed test_check_types_with_one_non_matching_element. Retrieved 4/7 statements.
# Partially parsed test_check_types_with_custom_exception_type. Retrieved 3/8 statements.
# Partially parsed test_check_types_with_type_strings. Retrieved 5/6 statements.
# Partially parsed test_check_types_with_mixed_type_objects_and_strings. Retrieved 4/6 statements.
# Partially parsed test_check_types_with_empty_iterable. Retrieved 1/3 statements.
# Partially parsed test_check_types_with_non_iterable_source_class. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = None

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 3.14
    var_3 = [var_0, var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'builtins.int'
    var_4 = [var_3]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = 'builtins.str'

def test_case_0():
    var_0 = []

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_with_default_format. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 8/12 statements.
# Partially parsed test_serialize_with_format_argument. Retrieved 8/12 statements.
# Partially parsed test_serialize_empty_set. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_complex_serializer. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = {var_0, var_1, var_2}

def test_case_0():
    var_0 = lambda format, v: v * 2
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = 4
    var_6 = 6
    var_7 = {var_2, var_5, var_6}

def test_case_0():
    var_0 = lambda format, v: f'{format}:{v}'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1, var_2]
    var_4 = 'prefix'
    var_5 = 'prefix:a'
    var_6 = 'prefix:b'
    var_7 = {var_5, var_6}

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



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_correct_types. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/11 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_type_key. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_type_value. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_union_types_key. Retrieved 6/13 statements.
# Partially parsed test_constructor_with_union_types_value. Retrieved 6/13 statements.
# Partially parsed test_constructor_with_checkedpmap_instance. Retrieved 2/7 statements.
# Partially parsed test_constructor_repr. Retrieved 3/8 statements.
# Partially parsed test_constructor_str. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 2.25
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 3
    var_2 = 3.14
    var_3 = {var_0: var_0, var_1: var_2}

def test_case_0():
    var_0 = 'a'
    var_1 = 1.0
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
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 3
    var_5 = 2.0
    var_6 = 4.0
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v >= 0, 'Value negative')
    var_2 = [var_0, var_1]
    var_3 = -1
    var_4 = 2.0
    var_5 = {var_3: var_4}
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Key negative'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 2.5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 'text'
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = {var_0: var_0}

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

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_valid_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_with_invalid_key_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invalid_value_type. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_valid_invariant. Retrieved 4/10 statements.
# Partially parsed test_constructor_with_multiple_invariants. Retrieved 6/12 statements.
# Partially parsed test_constructor_with_multiple_invariants_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checked_type_key. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_checked_type_value. Retrieved 1/10 statements.
# Partially parsed test_constructor_with_union_key_type. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_union_value_type. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_self_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_empty_dict_and_size. Retrieved 2/8 statements.
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
    var_2 = 2
    var_3 = {var_1: var_1, var_2: var_2}

def test_case_0():
    var_0 = lambda k, v: (k >= 0, 'Key negative')
    var_1 = lambda k, v: (v > 0, 'Value non-positive')
    var_2 = [var_0, var_1]
    var_3 = 1
    var_4 = 2
    var_5 = 1.5
    var_6 = 2.25
    var_7 = {var_3: var_5, var_4: var_6}

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
    var_0 = 5

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 1.5
    var_3 = 'two'
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



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_dict_initial. Retrieved 5/11 statements.
# Partially parsed test_constructor_type_check_key. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_check. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_pass. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_checkedpmap_instance. Retrieved 3/8 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 3/8 statements.
# Partially parsed test_constructor_multiple_invariants. Retrieved 7/14 statements.
# Partially parsed test_constructor_checked_type_create_key. Retrieved 3/11 statements.
# Partially parsed test_constructor_checked_type_create_value. Retrieved 3/11 statements.
# Partially parsed test_constructor_checked_type_create_both. Retrieved 3/14 statements.
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
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

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
    var_7 = 'Key negative'
    var_8 = 1
    var_9 = 0.0
    var_10 = {var_8: var_9}
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'Value non-positive'

def test_case_0():
    var_0 = '1'
    var_1 = 10
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 100
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = '1'
    var_1 = 100
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 1
    var_1 = 1.5
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_CheckedType_constructor_creates_instance.
# Partially parsed test_CheckedType_constructor_slots_are_empty. Retrieved 1/3 statements.
# Partially parsed test_CheckedType_constructor_has_required_methods. Retrieved 2/5 statements.
# Failed to parse test_CheckedType_constructor_create_is_classmethod.
# Failed to parse test_CheckedType_constructor_serialize_is_method.


def test_case_0():
    var_0 = []
    var_1 = '__dict__'

def test_case_0():
    var_0 = []
    var_1 = 'create'
    var_2 = 'serialize'



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_check_types_with_expected_types_and_matching_element.
# Failed to parse test_check_types_with_expected_types_and_non_matching_element.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 1/11 statements.
# Partially parsed test_check_types_with_none_expected_types. Retrieved 1/11 statements.
# Failed to parse test_check_types_with_multiple_expected_types_and_matching_element.
# Failed to parse test_check_types_with_multiple_expected_types_and_non_matching_element.
# Failed to parse test_check_types_with_iterator_and_matching_element.
# Failed to parse test_check_types_with_multiple_elements_and_matching_elements.
# Failed to parse test_check_types_with_multiple_elements_and_one_non_matching_element.
# Failed to parse test_check_types_with_default_exception_type.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = bool(True)
    assert var_2 is True
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_types_with_expected_types_and_matching_element. Retrieved 1/12 statements.
# Partially parsed test_check_types_with_expected_types_and_non_matching_element. Retrieved 1/15 statements.
# Partially parsed test_check_types_with_empty_expected_types. Retrieved 2/12 statements.
# Partially parsed test_check_types_with_none_expected_types. Retrieved 2/12 statements.
# Partially parsed test_check_types_with_multiple_expected_types_and_matching_element. Retrieved 1/14 statements.
# Partially parsed test_check_types_with_multiple_expected_types_and_non_matching_element. Retrieved 1/17 statements.
# Partially parsed test_check_types_with_multiple_elements_all_matching. Retrieved 1/14 statements.
# Partially parsed test_check_types_with_multiple_elements_one_non_matching. Retrieved 1/17 statements.
# Partially parsed test_check_types_with_default_exception_type. Retrieved 1/12 statements.
# Partially parsed test_check_types_with_get_type_function. Retrieved 1/12 statements.


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

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = None

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

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'MockSourceClass'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = bool(True)
    assert var_2 is True

def test_case_0():
    var_0 = 'MockSourceClass'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test__restore_pickle. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'test_data'
    var_1 = set()



# Parsed testcases at query #18
#--------------------------

# Failed to parse test__checked_type_create_with_same_class.
# Partially parsed test__checked_type_create_with_checked_type_subclass. Retrieved 5/16 statements.
# Failed to parse test__checked_type_create_with_matching_type_in_list.
# Partially parsed test__checked_type_create_without_checked_type. Retrieved 4/9 statements.
# Partially parsed test__checked_type_create_with_ignore_extra. Retrieved 6/17 statements.


def test_case_0():
    var_0 = '__main__.MockType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = '__main__.MockType'
    var_1 = [var_0]
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = True



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_preserved_type.
# Failed to parse test_simple_type.
# Failed to parse test_iterable_of_types.
# Failed to parse test_nested_iterable.



def test_case_0():
    var_0 = 'int'
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(var_1 == ['int'])
    assert var_2 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.maybe_parse_user_type(var_0)
    var_2 = bool(False)
    assert var_2 is True
    var_3 = bool(True)
    assert var_3 is True



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_constructor_with_empty_initial.
# Partially parsed test_constructor_with_initial_dict. Retrieved 5/11 statements.
# Partially parsed test_constructor_type_check_key_violation. Retrieved 3/8 statements.
# Partially parsed test_constructor_type_check_value_violation. Retrieved 3/8 statements.
# Partially parsed test_constructor_invariant_violation. Retrieved 4/9 statements.
# Partially parsed test_constructor_invariant_satisfied. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_size_parameter. Retrieved 2/8 statements.
# Partially parsed test_constructor_with_checked_key_type_create. Retrieved 8/16 statements.
# Partially parsed test_constructor_with_checked_value_type_create. Retrieved 8/16 statements.
# Partially parsed test_constructor_repr. Retrieved 4/9 statements.
# Partially parsed test_constructor_str. Retrieved 4/9 statements.


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
    var_0 = {}
    var_1 = 0

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1
    var_6 = 2
    var_7 = {var_5: var_2, var_6: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 100
    var_3 = 200
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = '100'
    var_6 = '200'
    var_7 = {var_0: var_5, var_1: var_6}

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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_constructor_with_empty_initial. Retrieved 1/6 statements.
# Partially parsed test_constructor_with_list_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_set_initial. Retrieved 4/9 statements.
# Partially parsed test_constructor_with_tuple_initial. Retrieved 5/10 statements.
# Partially parsed test_constructor_with_pmap_initial. Retrieved 6/11 statements.
# Partially parsed test_constructor_type_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_invariant_check_failure. Retrieved 4/8 statements.
# Partially parsed test_constructor_duplicate_elements. Retrieved 4/9 statements.
# Partially parsed test_constructor_no_type_or_invariant. Retrieved 4/9 statements.


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
    var_1 = 4
    var_2 = 5
    var_3 = {var_1, var_2}
    var_4 = 4
    var_5 = 5

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 6
    var_2 = 7
    var_3 = 8
    var_4 = (var_1, var_2, var_3)
    var_5 = 6
    var_6 = 7
    var_7 = 8

import pyrsistent._pmap as module_0


def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 9
    var_2 = 10
    var_3 = True
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 9
    var_7 = 10

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 'a'
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = -1
    var_3 = [var_1, var_2]
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(True)
    assert var_5 is True

def test_case_0():
    var_0 = lambda n: (n >= 0, 'Negative')
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_1, var_2, var_2]
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_predicate_at_line_18_evaluates_to_true_for_type_and_not_iterable.




