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
    var_11 = bool(False)
    assert var_11 is True


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
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 500
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = module_0.check_global_invariants(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 6/11 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/7 statements.


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
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'default'
    var_1 = 'not_callable'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not_callable'
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = 'not_callable'
    var_5 = bool(False)
    assert var_5 is True

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


def test_case_0():
    var_0 = None
    var_1 = 'default'
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x
    var_6 = module_0.field(var_0, var_3, var_1, factory=var_4, serializer=var_5)
    var_7 = module_0._check_field_parameters(var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 9/17 statements.



def test_case_0():
    var_0 = 'int'
    var_1 = 'str'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = 'MockPMapField'
    var_6 = ()
    var_7 = {}
    var_8 = [var_5, var_6, var_7]
    var_9 = module_0._restore_pmap_field_pickle(var_0, var_1, var_4)
    assert var_9 == 'restored_object'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 7/9 statements.
# Partially parsed test_pfield_constructor_with_defaults. Retrieved 5/7 statements.


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
    var_0 = ()
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = False
    var_5 = None
    var_6 = lambda : var_5
    var_7 = lambda x: x
    var_8 = module_0._PField(var_0, var_2, var_3, var_4, var_6, var_7)
    var_9 = var_8.type
    var_10 = bool(var_8.type == var_0)
    assert var_10 is True
    var_11 = var_8.invariant
    var_12 = bool(var_8.invariant == var_2)
    assert var_12 is True
    var_13 = var_8.initial
    var_14 = bool(var_8.initial == var_3)
    assert var_14 is True
    var_15 = var_8.mandatory
    var_16 = bool(var_8.mandatory == var_4)
    assert var_16 is True
    var_17 = var_8._factory
    var_18 = bool(var_8._factory == var_6)
    assert var_18 is True
    var_19 = var_8.serializer
    var_20 = bool(var_8.serializer == var_7)
    assert var_20 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_custom_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 5/10 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 5/10 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_no_type. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_invariant. Retrieved 3/8 statements.
# Partially parsed test_check_field_parameters_invalid_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_invalid_serializer. Retrieved 4/9 statements.


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
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

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
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x

def test_case_0():
    var_0 = 'not callable'
    var_1 = lambda x: x
    var_2 = lambda x: x
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 'not callable'
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = 'not callable'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_false_when_ignore_extra_false. Retrieved 9/12 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_when_not_type_cls. Retrieved 7/12 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_when_no_ignore_extra_param. Retrieved 5/13 statements.
# Partially parsed test_is_field_ignore_extra_complaint_true_when_all_conditions_met. Retrieved 5/13 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_with_empty_type_set. Retrieved 9/12 statements.
# Partially parsed test_is_field_ignore_extra_complaint_true_with_string_type. Retrieved 7/14 statements.


def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = set()
    var_5 = None
    var_6 = lambda : var_5
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = False

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = True

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = True

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = True

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = set()
    var_5 = None
    var_6 = lambda : var_5
    var_7 = {var_2: var_4, var_3: var_6}
    var_8 = [var_0, var_1, var_7]
    var_9 = True

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'builtins.int'
    var_5 = {var_4}
    var_6 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_factory_property_with_non_checked_type. Retrieved 3/9 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = None



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_field_parameters_field_initial_not_callable_and_type_mismatch. Retrieved 14/20 statements.


import collections as module_0


def test_case_0():
    var_0 = 'Field'
    var_1 = 'type'
    var_2 = 'initial'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 'not_an_int'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_no_type_restriction. Retrieved 8/16 statements.
# Partially parsed test_check_type_with_custom_class. Retrieved 1/11 statements.
# Partially parsed test_check_type_with_type_string. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'string'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field DestinationClass.test_field, was str'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 42
    var_2 = 'string'

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 42
    var_3 = 'string'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]

def test_case_0():
    var_0 = 'test_field'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'test_field'
    var_3 = 42



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 21/65 statements.



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



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_sets_correct_name.
# Failed to parse test_make_pmap_field_type_with_tuple_types.
# Failed to parse test_make_pmap_field_type_has_reduce_method.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_factory_property_with_non_checked_type. Retrieved 2/11 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_false_when_ignore_extra_false. Retrieved 3/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_when_not_type_cls. Retrieved 2/6 statements.
# Partially parsed test_is_field_ignore_extra_complaint_true_when_ignore_extra_in_factory. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_false_when_ignore_extra_not_in_factory. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_set_type. Retrieved 1/9 statements.
# Partially parsed test_is_field_ignore_extra_complaint_with_empty_tuple_type. Retrieved 3/6 statements.


def test_case_0():
    var_0 = None
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = None
    var_1 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = ()
    var_1 = None
    var_2 = True



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Partially parsed test_make_pmap_field_type_with_string_types. Retrieved 3/6 statements.
# Failed to parse test_make_pmap_field_type_name_formatting.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 5/11 statements.


import pyrsistent._field_common as module_0


def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)
    var_3 = var_2.__key_type__
    var_4 = var_2.__value_type__
    var_5 = var_2.__name__
    assert var_5 == 'IntToStrPMap'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #16
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = 1001
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

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 2/6 statements.
# Partially parsed test_make_seq_field_type_caches_type. Retrieved 2/7 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 2/6 statements.
# Partially parsed test_make_seq_field_type_reduce_returns_restore_function. Retrieved 2/7 statements.


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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 4/9 statements.
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



# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_field_with_single_type.
# Failed to parse test_field_with_multiple_types_list.
# Failed to parse test_field_with_multiple_types_tuple.
# Failed to parse test_field_with_multiple_types_set.
# Partially parsed test_field_with_initial. Retrieved 1/2 statements.
# Partially parsed test_field_initial_type_mismatch. Retrieved 1/3 statements.
# Failed to parse test_field_with_nested_iterable_types.
# Failed to parse test_field_with_preserved_iterable_type.
# Failed to parse test_field_factory_default_for_checkedtype.
# Partially parsed test_field_invariant_wrapped. Retrieved 2/10 statements.



def test_case_0():
    var_0 = 'int'
    var_1 = module_0.field(var_0)
    var_2 = var_1.type
    var_3 = bool(var_1.type == {'int'})
    assert var_3 is True


def test_case_0():
    var_0 = True
    var_1 = ''
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.field(invariant=var_3)
    var_5 = var_4.invariant
    var_6 = var_4.invariant
    var_7 = callable(var_6)
    var_8 = bool(var_7)
    assert var_8 is True

def test_case_0():
    var_0 = 10


def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = var_1.mandatory
    assert var_2 is True


def test_case_0():
    var_0 = 2
    var_1 = lambda x: x * var_0
    var_2 = module_0.field(factory=var_1)
    var_3 = var_2.factory
    var_4 = bool(var_2.factory == var_1)
    assert var_4 is True


def test_case_0():
    var_0 = lambda x: str(x)
    var_1 = module_0.field(serializer=var_0)
    var_2 = var_1.serializer
    var_3 = bool(var_1.serializer == var_0)
    assert var_3 is True


def test_case_0():
    var_0 = 123
    var_1 = module_0.field(var_0)
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = bool(False)
    assert var_1 is True


def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = 'not_callable'
    var_1 = module_0.field(serializer=var_0)
    var_2 = bool(False)
    assert var_2 is True


def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = var_0.type
    var_3 = bool(var_0.type == var_1)
    assert var_3 is True

def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 'bad'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_serialize_checked_type_with_pfield_no_serializer. Retrieved 1/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []



# Parsed testcases at query #23
#--------------------------





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


def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_predicate_at_line_6_false.




# Parsed testcases at query #26
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



# Parsed testcases at query #27
#--------------------------





def test_case_0():
    var_0 = False
    var_1 = 1001
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



# Parsed testcases at query #28
#--------------------------





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


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = (var_5, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True


def test_case_0():
    var_0 = 'test'
    var_1 = False
    var_2 = 100
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 200
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 300
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 14/20 statements.


import collections as module_0


def test_case_0():
    var_0 = 'Field'
    var_1 = 'type'
    var_2 = 'initial'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 'not_an_int'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = bool(False)
    assert var_15 is True



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_set_fields_with_base_containing_name. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_overlapping_keys_in_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield_in_dct. Retrieved 3/8 statements.
# Partially parsed test_set_fields_with_pfield_and_regular_key. Retrieved 5/10 statements.
# Partially parsed test_set_fields_with_base_and_pfield. Retrieved 3/10 statements.


import pyrsistent._field_common as module_0


def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'test': {}})
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'test'
    var_5 = bool(var_3 == {'test': {'a': 1}})
    assert var_5 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test'
    var_8 = bool(var_6 == {'test': {'a': 1, 'b': 2}})
    assert var_8 is True

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
    var_12 = bool(var_10 == {'test': {'a': 1, 'c': 4, 'b': 2}})
    assert var_12 is True

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'test'

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = []
    var_4 = 'test'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 'test'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_type_cls_check_fails. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 'some_type'
    var_1 = lambda : None
    var_2 = 'not_a_type_class'
    var_3 = True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
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
    var_0 = 123
    var_1 = 'yaml'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_fields_adds_name_dict. Retrieved 4/8 statements.
# Partially parsed test_set_fields_merges_base_dicts. Retrieved 4/8 statements.
# Partially parsed test_set_fields_moves_pfield. Retrieved 3/8 statements.
# Partially parsed test_set_fields_overwrites_existing_name. Retrieved 7/10 statements.
# Partially parsed test_set_fields_handles_multiple_pfields. Retrieved 4/10 statements.
# Partially parsed test_set_fields_mixed_base_and_pfield. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = {}
    var_3 = 'test_fields'
    var_4 = bool(var_2 == {'test_fields': {'field1': 'value1', 'field2': 'value2'}})
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_fields'
    var_8 = bool(var_6 == {'test_fields': {'a': 1, 'b': 2}})
    assert var_8 is True

def test_case_0():
    var_0 = 'custom'
    var_1 = ()
    var_2 = 'fields'


def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'meta'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'meta': {}})
    assert var_4 is True

def test_case_0():
    var_0 = 'old'
    var_1 = 'meta'
    var_2 = 'key'
    var_3 = 'original'
    var_4 = {var_2: var_3}
    var_5 = {var_1: var_4}
    var_6 = 'meta'
    var_7 = bool(var_5 == {'meta': {'existing': 'old'}})
    assert var_7 is True

def test_case_0():
    var_0 = 'fieldA'
    var_1 = 'fieldB'
    var_2 = ()
    var_3 = 'attrs'


def test_case_0():
    var_0 = 'regular'
    var_1 = 'function'
    var_2 = 42
    var_3 = lambda x: x
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = ()
    var_6 = 'special'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = bool(var_4 == {'special': {}})
    assert var_8 is True

def test_case_0():
    var_0 = 'base_value'
    var_1 = 'pfield'
    var_2 = 'fields'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_fields_pfield_condition_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'not_pfield'
    var_3 = []
    var_4 = 'test_name'
    var_5 = 'key1'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_check_global_invariants_subject_passed. Retrieved 2/8 statements.



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


def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 5
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 3
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)
    var_10 = bool(False)
    assert var_10 is True


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
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)

def test_case_0():
    var_0 = None
    var_1 = 'specific_subject'
    var_2 = bool(var_0 == var_1)
    assert var_2 is True



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_restore_seq_field_pickle.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_sequence_field_creates_checked_type. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_with_optional_true. Retrieved 6/13 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 5/10 statements.
# Partially parsed test_sequence_field_initial_factory_called. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_with_custom_invariant. Retrieved 5/12 statements.
# Partially parsed test_sequence_field_type_set_correctly. Retrieved 4/7 statements.
# Partially parsed test_sequence_field_optional_type_includes_none. Retrieved 5/11 statements.
# Partially parsed test_sequence_field_factory_creates_instance. Retrieved 6/13 statements.
# Partially parsed test_sequence_field_initial_none_when_optional. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_mandatory_always_true. Retrieved 4/8 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = [var_4]
    var_6 = [var_0, var_1, var_2]

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
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = {var_1, var_2}

def test_case_0():
    var_0 = True
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = None
    var_5 = [var_4]

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = 4
    var_3 = 5
    var_4 = 6
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = True
    var_3 = []



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_sequence_field_factory_for_optional_none. Retrieved 14/36 statements.


def test_case_0():
    var_0 = lambda self, *args, **kwargs: 'created'
    var_1 = True
    var_2 = None
    var_3 = 'TheType'
    var_4 = ()
    var_5 = 'create'
    var_6 = 'TheType_created'
    var_7 = lambda *args, **kwargs: var_6
    var_8 = {var_5: var_7}
    var_9 = [var_3, var_4, var_8]
    var_10 = None
    var_11 = lambda argument, **kwargs: var_10 if argument is var_10 else TheType.create(argument, **kwargs)
    var_12 = True
    var_13 = None
    var_14 = False



# Parsed testcases at query #8
#--------------------------





def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = False
    var_5 = (var_4, var_0)
    var_6 = lambda x: var_5
    var_7 = None
    var_8 = [var_6, var_6, var_6]
    var_9 = module_0.check_global_invariants(var_7, var_8)
    var_10 = bool(False)
    assert var_10 is True


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = [var_3, var_3]
    var_6 = module_0.check_global_invariants(var_4, var_5)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 3/7 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 4/9 statements.
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_checked_type_with_pfield_no_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_non_checked_type_with_pfield_no_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_checked_type_with_other_serializer. Retrieved 1/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = 'some_value'

def test_case_0():
    var_0 = 'json'
    var_1 = []



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_returns_cached_type. Retrieved 2/9 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = lambda x: x == var_0

def test_case_0():
    var_0 = 'test'
    var_1 = lambda x: x == var_0

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_multiple_types_valid_initial. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_multiple_types_invalid_initial. Retrieved 1/6 statements.


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


def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True


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

def test_case_0():
    var_0 = 12.34
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #13
#--------------------------





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


def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_serialize_checked_type_with_pfield_no_serializer. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_restore_pmap_field_pickle.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_parameter. Retrieved 1/6 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 1/5 statements.
# Partially parsed test_check_field_parameters_no_initial. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_callable_initial. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_multiple_types_valid_initial. Retrieved 5/9 statements.


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


def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True


def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0._check_field_parameters(var_1)
    var_3 = bool(False)
    assert var_3 is True


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



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.
# Partially parsed test_serialize_with_checked_type_and_no_serializer_returns_serialized. Retrieved 2/7 statements.
# Partially parsed test_serialize_with_checked_type_and_no_serializer_calls_serialize_with_format. Retrieved 2/7 statements.


def test_case_0():
    var_0 = None
    var_1 = 'json'

def test_case_0():
    var_0 = 'xml'


def test_case_0():
    var_0 = 'test_string'
    var_1 = None
    var_2 = 'json'
    var_3 = module_0.serialize(var_1, var_2, var_0)
    assert var_3 is None

def test_case_0():
    var_0 = 'test_string'
    var_1 = 'yaml'

def test_case_0():
    var_0 = None
    var_1 = 'binary'

def test_case_0():
    var_0 = None
    var_1 = 'csv'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_no_type_specified. Retrieved 3/9 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_multiple_invalid_types. Retrieved 2/10 statements.
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
    var_3 = 'Invalid type for field DestinationClass.field_name, was str'

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
    var_3 = 'Invalid type for field DestinationClass.field_name, was float'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'field_name'
    var_3 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'field_name'
    var_3 = 'not_an_int'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invalid type for field DestinationClass.field_name, was str'



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 13/28 statements.


import collections as module_0


def test_case_0():
    var_0 = []
    var_1 = 'Field'
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = [var_2, var_3, var_4, var_5, var_6]
    var_8 = module_0.namedtuple(var_1, var_7)
    var_9 = []
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = lambda x: x
    var_13 = lambda x: x



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 6/11 statements.
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
    var_1 = 5
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 'string'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

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
    var_1 = 5
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = 'not callable'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_check_field_parameters_valid_field. Retrieved 5/8 statements.
# Partially parsed test_check_field_parameters_invalid_type_element. Retrieved 6/11 statements.
# Partially parsed test_check_field_parameters_invalid_initial_type. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_non_callable_factory. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_non_callable_serializer. Retrieved 5/9 statements.
# Partially parsed test_check_field_parameters_callable_initial_with_type. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_pfield_no_initial. Retrieved 4/7 statements.
# Partially parsed test_check_field_parameters_multiple_types_valid_initial. Retrieved 5/9 statements.


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
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'default'
    var_1 = 'not callable'
    var_2 = lambda x: x
    var_3 = lambda x: x
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = 'not callable'
    var_4 = lambda x: x
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = 'not callable'
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = 'callable'
    var_1 = lambda : var_0
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: x
    var_5 = lambda x: x

import pyrsistent._field_common as module_0


def test_case_0():
    var_0 = 123
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x
    var_5 = module_0.field(invariant=var_2, initial=var_0, factory=var_3, serializer=var_4)
    var_6 = module_0._check_field_parameters(var_5)

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: x
    var_3 = lambda x: x

def test_case_0():
    var_0 = 10
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x
    var_4 = lambda x: x



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_different_types.
# Failed to parse test_make_pmap_field_type_with_custom_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_with_key_and_value_types.
# Partially parsed test_pmap_field_optional_allows_none. Retrieved 3/10 statements.
# Partially parsed test_pmap_field_invariant_is_wrapped. Retrieved 3/9 statements.
# Failed to parse test_pmap_field_initial_is_empty_checked_pmap.
# Partially parsed test_pmap_field_factory_creates_checked_pmap_from_dict. Retrieved 5/9 statements.
# Partially parsed test_pmap_field_with_custom_invariant_enforces_constraint. Retrieved 8/16 statements.
# Failed to parse test_pmap_field_mandatory_is_true.
# Partially parsed test_pmap_field_optional_factory_handles_none_and_dict. Retrieved 6/11 statements.
# Failed to parse test_pmap_field_type_set_contains_checked_pmap_class.
# Failed to parse test_pmap_field_serializer_is_default.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = {}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}
    var_3 = 2
    var_4 = 3
    var_5 = 'b'
    var_6 = 'c'
    var_7 = {var_0: var_1, var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'key'
    var_3 = 2
    var_4 = [var_0, var_3]
    var_5 = {var_2: var_4}



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_checked_type_with_pfield_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_checked_type_with_other_serializer. Retrieved 1/8 statements.
# Partially parsed test_serialize_non_checked_type_with_pfield_no_serializer. Retrieved 2/6 statements.
# Partially parsed test_serialize_non_checked_type_with_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []

def test_case_0():
    var_0 = 'xml'
    var_1 = []

def test_case_0():
    var_0 = []
    var_1 = 'yaml'
    var_2 = 42

def test_case_0():
    var_0 = 'csv'
    var_1 = 'data'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_check_type_valid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_no_type_specified. Retrieved 3/9 statements.
# Partially parsed test_check_type_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_multiple_invalid_type. Retrieved 2/10 statements.
# Partially parsed test_check_type_with_type_string. Retrieved 3/9 statements.
# Partially parsed test_check_type_with_type_string_invalid. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'field_name'
    var_1 = 42

def test_case_0():
    var_0 = 'field_name'
    var_1 = 'string'
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
    var_2 = 'string'

def test_case_0():
    var_0 = 'field_name'
    var_1 = 3.14
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Invalid type for field DestinationCls.field_name, was float'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'field_name'
    var_3 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = [var_0]
    var_2 = 'field_name'
    var_3 = 'string'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Invalid type for field DestinationCls.field_name, was str'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_ignore_extra_false. Retrieved 2/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_not_type_cls. Retrieved 1/4 statements.
# Partially parsed test_is_field_ignore_extra_complaint_no_ignore_extra_param. Retrieved 1/7 statements.
# Partially parsed test_is_field_ignore_extra_complaint_valid. Retrieved 1/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_empty_type_set. Retrieved 2/8 statements.
# Partially parsed test_is_field_ignore_extra_complaint_type_tuple. Retrieved 1/8 statements.


def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = set()
    var_1 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_make_seq_field_type_creates_subclass. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_name_generation. Retrieved 1/7 statements.
# Partially parsed test_make_seq_field_type_reuse_cached_type. Retrieved 2/8 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 1/7 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0

def test_case_0():
    var_0 = None



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_restore_pmap_field_pickle.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_set_fields_with_base_containing_name. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield_in_dct. Retrieved 3/8 statements.
# Partially parsed test_set_fields_with_bases_and_pfield. Retrieved 3/10 statements.
# Partially parsed test_set_fields_with_overlapping_keys_in_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_empty_dict_in_base. Retrieved 3/6 statements.
# Partially parsed test_set_fields_with_non_dict_in_base. Retrieved 3/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = bool(var_0 == {'test': {}})
    assert var_4 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 'test'
    var_5 = bool(var_3 == {'test': {'a': 1}})
    assert var_5 is True

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test'
    var_8 = bool(var_6 == {'test': {'a': 1, 'b': 2}})
    assert var_8 is True

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 'test'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 'test'

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
    var_12 = bool(var_10 == {'test': {'a': 1, 'c': 4, 'b': 2}})
    assert var_12 is True

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'test'
    var_3 = bool(var_1 == {'test': {}})
    assert var_3 is True

def test_case_0():
    var_0 = 'not a dict'
    var_1 = {}
    var_2 = 'test'
    var_3 = bool(var_1 == {'test': {}})
    assert var_3 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_checked_type_with_pfield_no_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_checked_type_with_other_serializer. Retrieved 1/9 statements.
# Partially parsed test_serialize_non_checked_type_with_pfield_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_non_checked_type_with_custom_serializer. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 'json'
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 'json'

def test_case_0():
    var_0 = 'json'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 1/6 statements.
# Partially parsed test_serialize_with_checked_type_and_custom_serializer. Retrieved 1/7 statements.
# Partially parsed test_serialize_with_non_checked_type_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_non_checked_type_and_custom_serializer. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 'json'

def test_case_0():
    var_0 = 'xml'

def test_case_0():
    var_0 = 'some_value'
    var_1 = 'json'

def test_case_0():
    var_0 = 'some_value'
    var_1 = 'yaml'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_make_seq_field_type_creates_new_type. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_caches_type. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_sets_name. Retrieved 1/5 statements.
# Partially parsed test_make_seq_field_type_reduce_method. Retrieved 1/6 statements.
# Partially parsed test_make_seq_field_type_with_multiple_checked_types. Retrieved 1/5 statements.


def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = None



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.
# Partially parsed test_pfield_constructor_with_defaults. Retrieved 5/7 statements.
# Partially parsed test_pfield_constructor_with_multiple_types. Retrieved 5/7 statements.
# Partially parsed test_pfield_constructor_with_complex_invariant. Retrieved 5/9 statements.


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


def test_case_0():
    var_0 = ()
    var_1 = None
    var_2 = None
    var_3 = False
    var_4 = None
    var_5 = None
    var_6 = module_0._PField(var_0, var_1, var_2, var_3, var_4, var_5)
    var_7 = var_6.type
    var_8 = bool(var_6.type == var_0)
    assert var_8 is True
    var_9 = var_6.invariant
    var_10 = bool(var_6.invariant == var_1)
    assert var_10 is True
    var_11 = var_6.initial
    var_12 = bool(var_6.initial == var_2)
    assert var_12 is True
    var_13 = var_6.mandatory
    var_14 = bool(var_6.mandatory == var_3)
    assert var_14 is True
    var_15 = var_6._factory
    var_16 = bool(var_6._factory == var_4)
    assert var_16 is True
    var_17 = var_6.serializer
    var_18 = bool(var_6.serializer == var_5)
    assert var_18 is True

def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = True
    var_3 = None
    var_4 = None

def test_case_0():
    var_0 = 0
    var_1 = lambda lst: len(lst) > var_0
    var_2 = []
    var_3 = False
    var_4 = ','



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 14/20 statements.


import collections as module_0


def test_case_0():
    var_0 = 'Field'
    var_1 = 'type'
    var_2 = 'initial'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 'not_an_int'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Initial has invalid type'



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_predicate_at_line_6_evaluates_to_false.




# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
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
    var_11 = bool(False)
    assert var_11 is True


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
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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



# Parsed testcases at query #23
#--------------------------





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


def test_case_0():
    var_0 = 'test_subject'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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


def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 999
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = [var_4]
    var_6 = module_0.check_global_invariants(var_0, var_5)
    var_7 = bool(False)
    assert var_7 is True


def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 300
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 200
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = 100
    var_9 = (var_1, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_4, var_7, var_10]
    var_12 = module_0.check_global_invariants(var_0, var_11)
    var_13 = bool(False)
    assert var_13 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 4/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #25
#--------------------------





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



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_check_field_parameters_initial_invalid_type. Retrieved 14/20 statements.


import collections as module_0


def test_case_0():
    var_0 = 'Field'
    var_1 = 'type'
    var_2 = 'initial'
    var_3 = 'invariant'
    var_4 = 'factory'
    var_5 = 'serializer'
    var_6 = [var_1, var_2, var_3, var_4, var_5]
    var_7 = module_0.namedtuple(var_0, var_6)
    var_8 = []
    var_9 = 'not_an_int'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'Initial has invalid type'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_set_fields_pfield_condition_true. Retrieved 5/10 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'normal_value'
    var_3 = []
    var_4 = 'test_name'
    var_5 = 'field1'



# Parsed testcases at query #28
#--------------------------




import pyrsistent._field_common as module_0


def test_case_0():
    var_0 = 'error1'
    var_1 = 'error2'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = (var_3, var_0)
    var_5 = lambda x: var_4
    var_6 = (var_3, var_1)
    var_7 = lambda x: var_6
    var_8 = [var_5, var_7]
    var_9 = 'test_subject'
    var_10 = module_0.check_global_invariants(var_9, var_8)
    var_11 = bool(False)
    assert var_11 is True



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_sets_correct_name.
# Failed to parse test_make_pmap_field_type_with_tuple_types.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = {var_0: var_1}



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_check_field_parameters_predicate_false. Retrieved 2/17 statements.


def test_case_0():
    var_0 = True
    var_1 = 0



# Parsed testcases at query #31
#--------------------------






# Parsed testcases at query #32
#--------------------------





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
    var_11 = bool(False)
    assert var_11 is True


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
    var_13 = bool(False)
    assert var_13 is True


def test_case_0():
    var_0 = 'test'
    var_1 = []
    var_2 = module_0.check_global_invariants(var_0, var_1)


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



# Parsed testcases at query #33
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_cached_class.
# Failed to parse test_make_pmap_field_type_with_custom_class_name.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 6/14 statements.


def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 1.5
    var_3 = 2.5
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 2



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_set_fields_pfield_assignment. Retrieved 6/18 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = 'key2'
    var_3 = 'value2'
    var_4 = 'extra_key'
    var_5 = 'test_name'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/60 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = True
    var_4 = lambda x: str(x)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_factory_property_with_checked_type. Retrieved 2/15 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = True



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_set_fields_pfield_condition_true. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'key1'
    var_1 = 'key2'
    var_2 = 'not_pfield'
    var_3 = []
    var_4 = 'test_name'
    var_5 = 'test_name'
    var_6 = 'test_name'
    var_7 = 'key1'
    var_8 = 'key1'
    var_9 = 'key2'



# Parsed testcases at query #39
#--------------------------





def test_case_0():
    var_0 = 'ERROR_1'
    var_1 = 'ERROR_2'
    var_2 = (var_0, var_1)
    var_3 = False
    var_4 = var_2[var_3]
    var_5 = (var_3, var_4)
    var_6 = lambda s: var_5
    var_7 = 1
    var_8 = var_2[var_7]
    var_9 = (var_3, var_8)
    var_10 = lambda s: var_9
    var_11 = [var_6, var_10]
    var_12 = None
    var_13 = module_0.check_global_invariants(var_12, var_11)
    var_14 = False
    var_15 = True
    var_16 = bool(var_15)
    assert var_16 is True


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda s: var_2
    var_4 = [var_3]
    var_5 = None
    var_6 = module_0.check_global_invariants(var_5, var_4)



