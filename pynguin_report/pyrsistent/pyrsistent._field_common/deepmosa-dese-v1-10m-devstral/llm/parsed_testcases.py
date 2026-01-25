####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]
    var_8 = module_1.check_global_invariants(var_0, var_7)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'E1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = [var_4, var_8]
    var_10 = module_1.check_global_invariants(var_0, var_9)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'E1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'E2'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = module_1.check_global_invariants(var_0, var_8)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 3/4 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.


import pyrsistent._checked_types as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'format'
    var_2 = module_1.serialize(var_1)

def test_case_0():
    var_0 = 'format'
    var_1 = 'test_value'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_field. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 3/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = lambda x: True
    var_1 = lambda : None
    var_2 = lambda x: x

def test_case_0():
    var_0 = 5.5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = 'not callable'
    var_2 = lambda : None
    var_3 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = 'not callable'
    var_3 = lambda x: x

def test_case_0():
    var_0 = 5
    var_1 = lambda x: True
    var_2 = lambda : None
    var_3 = 'not callable'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pmap_field_creates_correct_field_with_non_optional. Retrieved 1/11 statements.
# Partially parsed test_pmap_field_creates_correct_field_with_optional. Retrieved 5/18 statements.
# Partially parsed test_pmap_field_with_custom_invariant. Retrieved 1/8 statements.


def test_case_0():
    var_0 = False

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}

def test_case_0():
    var_0 = False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_set_fields_with_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_with_pfield. Retrieved 10/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'fields'
    var_7 = module_0.set_fields(var_4, var_5, var_6)

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'y'
    var_6 = 'z'
    var_7 = 3
    var_8 = 4
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = {}
    var_11 = 'fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0._PField()
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'fields'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = 'fields'
    var_9 = var_4[var_8][var_0]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 14/19 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 123
    var_8 = [var_7]
    var_9 = True
    var_10 = lambda x: var_9
    var_11 = None
    var_12 = lambda : var_11
    var_13 = lambda x: x



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 1/8 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_set_fields_single_base_with_items. Retrieved 4/9 statements.
# Partially parsed test_set_fields_multiple_bases_with_items. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)

def test_case_0():
    var_0 = {}
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = 'test_name'

def test_case_0():
    var_0 = {}
    var_1 = 'key1'
    var_2 = 'value1'
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = 'test_name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = module_0._PField(var_1)
    var_3 = {var_0: var_2}
    var_4 = []
    var_5 = 'test_name'
    var_6 = module_0.set_fields(var_3, var_4, var_5)
    var_7 = module_0._PField(var_1)
    var_8 = {var_0: var_7}
    var_9 = {var_5: var_8}



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_subclass.
# Failed to parse test_make_pmap_field_type_reuses_existing_subclass.
# Failed to parse test_make_pmap_field_type_correct_name_generation.
# Partially parsed test_make_pmap_field_type_reduce_method. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test__sequence_field_with_checked_class_and_item_type. Retrieved 3/6 statements.
# Partially parsed test__sequence_field_with_optional. Retrieved 3/7 statements.
# Partially parsed test__sequence_field_with_invariant. Retrieved 4/6 statements.
# Partially parsed test__sequence_field_with_item_invariant. Retrieved 4/6 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = False
    var_3 = []

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = False
    var_3 = []



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_isinstance_check. Retrieved 7/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0._PField()
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'test_name'
    var_5 = module_0.set_fields(var_2, var_3, var_4)
    var_6 = var_2[var_0]



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 15/22 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'MockItemType'
    var_4 = ()
    var_5 = {}
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'MockType'
    var_11 = ()
    var_12 = 'create'
    var_13 = lambda self, data, **kwargs: data
    var_14 = {var_12: var_13}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)



# Parsed testcases at query #14
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)
    assert var_6 is None



# Parsed testcases at query #15
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'E001'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'E001'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'E002'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = module_1.check_global_invariants(var_0, var_8)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = False
    var_6 = 'E001'
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = (var_1, var_2)
    var_10 = lambda _: var_9
    var_11 = [var_4, var_8, var_10]
    var_12 = module_1.check_global_invariants(var_0, var_11)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_check_field_parameters_with_non_type_non_str_type. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 123
    var_1 = [var_0]



# Parsed testcases at query #17
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0._PField()
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'test'
    var_7 = module_0.set_fields(var_4, var_5, var_6)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_sequence_field_creates_checked_class_with_correct_type. Retrieved 2/5 statements.
# Partially parsed test_sequence_field_with_optional_creates_correct_factory. Retrieved 3/7 statements.
# Partially parsed test_sequence_field_without_optional_uses_default_factory. Retrieved 2/3 statements.
# Partially parsed test_sequence_field_sets_mandatory_to_true. Retrieved 2/3 statements.
# Partially parsed test_sequence_field_preserves_invariant. Retrieved 3/2 statements.
# Partially parsed test_sequence_field_creates_initial_value. Retrieved 6/10 statements.
# Partially parsed test_sequence_field_with_optional_type. Retrieved 3/5 statements.


def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = True
    var_1 = set()
    var_2 = None

def test_case_0():
    var_0 = False
    var_1 = []

def test_case_0():
    var_0 = False
    var_1 = set()

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = []

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1, var_2, var_3}
    var_5 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = True
    var_1 = []
    var_2 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'not_an_int'
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pmap_field_optional_false. Retrieved 1/2 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #22
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_name. Retrieved 2/5 statements.
# Partially parsed test__make_seq_field_type_creates_subclass_with_type_and_invariant. Retrieved 4/9 statements.
# Partially parsed test__make_seq_field_type_returns_cached_type. Retrieved 3/7 statements.
# Partially parsed test__make_seq_field_type_creates_subclass_with_reduce_method. Retrieved 5/10 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 5
    var_3 = -1

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 2
    var_3 = 3
    var_4 = [var_0, var_2, var_3]



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_reuses_existing_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'collections.OrderedDict'
    var_1 = 'typing.List'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_attributes. Retrieved 2/7 statements.
# Partially parsed test__make_seq_field_type_caches_created_type. Retrieved 2/7 statements.
# Partially parsed test__make_seq_field_type_uses_types_to_names_for_naming. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_reduce_returns_correct_tuple. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = 'not_callable'
    var_2 = None
    var_3 = lambda : var_2
    var_4 = lambda x: x



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 2/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'some_format'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test__make_seq_field_type_with_existing_type. Retrieved 4/9 statements.
# Partially parsed test__make_seq_field_type_with_new_type. Retrieved 2/9 statements.
# Partially parsed test__make_seq_field_type_with_string_type. Retrieved 3/9 statements.
# Partially parsed test__make_seq_field_type_with_custom_class. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 'TestType'
    var_3 = {}

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 0
    var_2 = lambda x: x > var_1

def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_predicate_at_line_3_evaluates_to_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #29
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)
    assert var_6 is None



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_restore_seq_field_pickle_with_valid_data. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]
    var_4 = 'MockClass'
    var_5 = ()
    var_6 = 'create'
    var_7 = lambda self, data, _factory_fields: data
    var_8 = {var_6: var_7}



# Parsed testcases at query #31
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_valid_field_parameters. Retrieved 6/16 statements.
# Partially parsed test_invalid_type_parameter. Retrieved 2/12 statements.
# Partially parsed test_invalid_initial_type. Retrieved 1/12 statements.
# Partially parsed test_non_callable_invariant. Retrieved 1/11 statements.
# Partially parsed test_non_callable_factory. Retrieved 1/11 statements.
# Partially parsed test_non_callable_serializer. Retrieved 1/11 statements.


def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 123
    var_1 = [var_0]

def test_case_0():
    var_0 = 'not an int'

def test_case_0():
    var_0 = 'not callable'

def test_case_0():
    var_0 = 'not callable'

def test_case_0():
    var_0 = 'not callable'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_restore_pmap_field_pickle_calls_restore_pickle_with_correct_args. Retrieved 10/14 statements.


def test_case_0():
    var_0 = 'MockType'
    var_1 = ()
    var_2 = 'create'
    var_3 = lambda data, _factory_fields: data
    var_4 = {var_2: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 16/25 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'MockItemType'
    var_4 = ()
    var_5 = {}
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'MockType'
    var_11 = ()
    var_12 = 'create'
    var_13 = lambda self, data, **kwargs: data
    var_14 = {var_12: var_13}
    var_15 = lambda cls, data: data



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_set_fields_predicate. Retrieved 6/12 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 'key1'
    var_2 = 'test_field'
    var_3 = 'key2'
    var_4 = 'test_field'
    var_5 = 'other_field'
    var_6 = {}
    var_7 = 'value'
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'test_field'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 3/10 statements.
# Partially parsed test__make_seq_field_type_returns_cached_type. Retrieved 3/10 statements.
# Partially parsed test__make_seq_field_type_with_different_types. Retrieved 4/12 statements.


def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = 5

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0
    var_2 = lambda x: var_0

def test_case_0():
    var_0 = 0
    var_1 = lambda x: len(x) > var_0
    var_2 = ''
    var_3 = 'test'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 2/12 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #38
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_type'
    var_1 = 'test_invariant'
    var_2 = 'test_initial'
    var_3 = True
    var_4 = 'test_factory'
    var_5 = 'test_serializer'
    var_6 = module_0._PField(var_0, var_1, var_2, var_3, var_4, var_5)



# Parsed testcases at query #39
#--------------------------

# Failed to parse test_check_field_parameters_valid_field.
# Failed to parse test_check_field_parameters_invalid_type_parameter.
# Failed to parse test_check_field_parameters_invalid_initial_type.
# Failed to parse test_check_field_parameters_non_callable_invariant.
# Failed to parse test_check_field_parameters_non_callable_factory.
# Failed to parse test_check_field_parameters_non_callable_serializer.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pmap_field_predicate_false. Retrieved 1/5 statements.


def test_case_0():
    var_0 = False



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_predicate_at_line_6_evaluates_to_false. Retrieved 6/27 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = lambda : var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x



# Parsed testcases at query #42
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_type'
    var_1 = 'test_invariant'
    var_2 = 'test_initial'
    var_3 = True
    var_4 = 'test_factory'
    var_5 = 'test_serializer'
    var_6 = module_0._PField(var_0, var_1, var_2, var_3, var_4, var_5)



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_type.
# Failed to parse test_make_pmap_field_type_reuses_existing_type.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/4 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'json'

def test_case_0():
    var_0 = 'xml'
    var_1 = 'data'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_make_pmap_field_type_with_builtin_types.
# Failed to parse test_make_pmap_field_type_caching.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'builtins.str'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_sequence_field_creates_checked_class_with_non_optional_type. Retrieved 3/11 statements.
# Partially parsed test_sequence_field_creates_checked_class_with_optional_type. Retrieved 4/11 statements.
# Partially parsed test_sequence_field_with_custom_invariant. Retrieved 6/11 statements.
# Partially parsed test_sequence_field_with_item_invariant. Retrieved 6/13 statements.


def test_case_0():
    var_0 = False
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = True
    var_1 = set()
    var_2 = None
    var_3 = set()

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = {var_1, var_2, var_3}
    var_5 = {var_1, var_2, var_3}



# Parsed testcases at query #5
#--------------------------




import pyrsistent._checked_types as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = None
    var_2 = 'test_format'
    var_3 = module_1.serialize(var_1, var_2, var_0)
    assert var_3 == 'serialized_test_format'



# Parsed testcases at query #6
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = True
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = False
    var_6 = 'error2'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_0.check_global_invariants(var_0, var_9)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test_subject'
    var_1 = False
    var_2 = 'error1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'error2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_0.check_global_invariants(var_0, var_8)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_field. Retrieved 6/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 7/10 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 6/10 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = [var_0]
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = None
    var_5 = lambda : var_4
    var_6 = lambda x: x

def test_case_0():
    var_0 = 42.0
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = None
    var_2 = lambda : var_1
    var_3 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: x

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 123



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_field.
# Partially parsed test_pmap_field_with_optional_creates_optional_field. Retrieved 5/16 statements.
# Partially parsed test_pmap_field_with_invariant_applies_invariant. Retrieved 5/8 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.wrap_invariant(var_3)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_valid_field_parameters. Retrieved 6/16 statements.
# Partially parsed test_invalid_type_parameter. Retrieved 6/17 statements.
# Partially parsed test_invalid_initial_type. Retrieved 6/17 statements.
# Partially parsed test_non_callable_invariant. Retrieved 4/15 statements.
# Partially parsed test_non_callable_factory. Retrieved 4/15 statements.
# Partially parsed test_non_callable_serializer. Retrieved 5/16 statements.


def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 3.14
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3
    var_5 = lambda x: str(x)

def test_case_0():
    var_0 = 42
    var_1 = None
    var_2 = lambda : var_1
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 42
    var_1 = True
    var_2 = lambda x: var_1
    var_3 = None
    var_4 = lambda : var_3



# Parsed testcases at query #11
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR_CODE'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = module_1.check_global_invariants(var_0, var_8)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = False
    var_6 = 'ERROR'
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = (var_1, var_2)
    var_10 = lambda _: var_9
    var_11 = [var_4, var_8, var_10]
    var_12 = module_1.check_global_invariants(var_0, var_11)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_sequence_field_with_checked_pset. Retrieved 9/19 statements.
# Partially parsed test_sequence_field_with_checked_pvector. Retrieved 8/18 statements.
# Partially parsed test_sequence_field_with_optional. Retrieved 6/15 statements.


def test_case_0():
    var_0 = False
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = True
    var_8 = lambda x: var_7

def test_case_0():
    var_0 = False
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = [var_1, var_2, var_3]
    var_5 = True
    var_6 = lambda x: var_5
    var_7 = lambda x: var_5

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = True
    var_3 = lambda x: var_2
    var_4 = lambda x: var_2
    var_5 = None



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_name. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_attributes. Retrieved 2/6 statements.
# Partially parsed test__make_seq_field_type_creates_subclass_with_correct_reduce. Retrieved 5/11 statements.
# Partially parsed test__make_seq_field_type_returns_cached_type. Retrieved 2/7 statements.


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

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_check_type_with_valid_type. Retrieved 2/8 statements.
# Partially parsed test_check_type_with_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_check_type_with_no_type_specified. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_multiple_valid_types. Retrieved 3/11 statements.
# Partially parsed test_check_type_with_string_type_name. Retrieved 3/8 statements.
# Partially parsed test_check_type_with_invalid_string_type_name. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = 123

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'not_an_int'

def test_case_0():
    var_0 = None
    var_1 = 'test_field'
    var_2 = 'any_value'

def test_case_0():
    var_0 = 'test_field'
    var_1 = 123
    var_2 = 'string'

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 456

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = 'test_field'
    var_3 = 'not_an_int'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = 42
    var_5 = lambda : var_4



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 2/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'some_format'



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_pmap_field_with_non_optional_and_no_invariant.
# Partially parsed test_pmap_field_with_optional_and_no_invariant. Retrieved 5/17 statements.
# Partially parsed test_pmap_field_with_non_optional_and_invariant. Retrieved 3/15 statements.
# Partially parsed test_pmap_field_with_optional_and_invariant. Retrieved 6/21 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}
    var_5 = {var_2: var_0}



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_fields_with_bases. Retrieved 4/8 statements.
# Partially parsed test_set_fields_combined. Retrieved 12/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'key2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = module_0._PField()
    var_3 = 'value2'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'test_name'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = module_0._PField()
    var_9 = {var_0: var_8}
    var_10 = {var_6: var_9, var_1: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'key1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = module_0._PField()
    var_6 = 'value2'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 'test_name'
    var_9 = 'key1'
    var_10 = 'value1'
    var_11 = module_0._PField()
    var_12 = {var_9: var_10, var_3: var_11}
    var_13 = {var_8: var_12, var_4: var_6}



# Parsed testcases at query #19
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 2/10 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 4/4 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 5/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {var_1}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {var_1}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_sequence_field_invariant_default. Retrieved 2/3 statements.


def test_case_0():
    var_0 = False
    var_1 = []



# Parsed testcases at query #21
#--------------------------




import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0._PField()
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'fields'
    var_7 = module_0.set_fields(var_4, var_5, var_6)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 6/13 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = set()



# Parsed testcases at query #23
#--------------------------

# Failed to parse test__make_pmap_field_type_creates_new_type.




# Parsed testcases at query #24
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)
    assert var_6 is None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_sequence_field_with_optional_true. Retrieved 7/15 statements.


def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = [var_1, var_2, var_3]
    var_6 = None



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_make_pmap_field_type_creates_new_class.
# Failed to parse test_make_pmap_field_type_returns_existing_class.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'builtins.str'
    var_1 = 'builtins.int'
    var_2 = module_0._make_pmap_field_type(var_0, var_1)



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/15 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 5/10 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.wrap_invariant(var_3)



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pmap_field_optional_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = False
    var_1 = None



# Parsed testcases at query #29
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda x: var_5
    var_7 = [var_4, var_6]
    var_8 = module_1.check_global_invariants(var_0, var_7)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = [var_4, var_8]
    var_10 = module_1.check_global_invariants(var_0, var_9)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda x: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda x: var_6
    var_8 = [var_4, var_7]
    var_9 = module_1.check_global_invariants(var_0, var_8)



# Parsed testcases at query #30
#--------------------------

# Partially parsed test__make_seq_field_type_creates_new_type. Retrieved 3/11 statements.
# Partially parsed test__make_seq_field_type_returns_cached_type. Retrieved 2/8 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = True
    var_2 = lambda x: var_1

def test_case_0():
    var_0 = True
    var_1 = lambda x: var_0



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pmap_field_optional_factory_returns_none. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/13 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 6/4 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 9/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_2: var_0}
    var_4 = {var_2: var_0}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = 'a'
    var_6 = {var_5: var_3}
    var_7 = {var_5: var_3}
    var_8 = {var_5: var_3}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = 'a'
    var_6 = {var_5: var_3}
    var_7 = {var_5: var_3}
    var_8 = {var_5: var_3}



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pmap_field_optional_predicate. Retrieved 2/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_set_fields_predicate. Retrieved 10/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0._PField()
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = []
    var_6 = 'test_name'
    var_7 = module_0.set_fields(var_4, var_5, var_6)
    var_8 = 'test_name'
    var_9 = var_4[var_8][var_0]



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 17/25 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'MockItemType'
    var_4 = ()
    var_5 = {}
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'MockType'
    var_11 = ()
    var_12 = 'create'
    var_13 = lambda self, data, _factory_fields: data
    var_14 = {var_12: var_13}
    var_15 = set()
    var_16 = lambda cls, data: cls.create(data, _factory_fields=var_15)



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_check_global_invariants_raises_exception_when_invariants_fail. Retrieved 1/8 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pfield_constructor. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = None



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_ignore_extra_is_false. Retrieved 2/3 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_field_type_is_not_a_subset_of_type_cls. Retrieved 7/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_false_when_field_factory_does_not_have_ignore_extra_param. Retrieved 7/11 statements.
# Partially parsed test_is_field_ignore_extra_complaint_returns_true_when_all_conditions_are_met. Retrieved 7/11 statements.


def test_case_0():
    var_0 = None
    var_1 = False

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda : var_4
    var_6 = True

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = None
    var_5 = lambda ignore_extra=False: var_4
    var_6 = True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_set_fields_predicate. Retrieved 10/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'fields'
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = {var_0: var_3}
    var_5 = 'fields'
    var_6 = 'b'
    var_7 = 2
    var_8 = {var_6: var_7}
    var_9 = {var_5: var_8}
    var_10 = 'fields'
    var_11 = 'c'
    var_12 = {}
    var_13 = 'test'
    var_14 = module_0._PField(var_13)
    var_15 = {var_10: var_12, var_11: var_14}
    var_16 = 'fields'
    var_17 = var_15[var_10][var_11]



# Parsed testcases at query #40
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/15 statements.
# Failed to parse test_pmap_field_with_invariant.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_2}



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_set_fields_with_bases. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'test_name'
    var_3 = module_0.set_fields(var_0, var_1, var_2)

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = 'test_name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = module_0._PField()
    var_2 = {var_0: var_1}
    var_3 = []
    var_4 = 'test_name'
    var_5 = module_0.set_fields(var_2, var_3, var_4)
    var_6 = module_0._PField()
    var_7 = {var_0: var_6}
    var_8 = {var_4: var_7}



# Parsed testcases at query #42
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR_CODE'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)



# Parsed testcases at query #43
#--------------------------

# Failed to parse test_pmap_field_basic.
# Partially parsed test_pmap_field_optional. Retrieved 5/16 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 4/4 statements.
# Partially parsed test_pmap_field_optional_with_invariant. Retrieved 8/4 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'a'
    var_3 = {var_0: var_2}
    var_4 = {var_0: var_2}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {var_1}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = {var_1}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = 'a'
    var_6 = {var_3: var_5}
    var_7 = {var_3: var_5}

def test_case_0():
    var_0 = True
    var_1 = 'test'
    var_2 = (var_0, var_1)
    var_3 = True
    var_4 = None
    var_5 = 'a'
    var_6 = {var_3: var_5}
    var_7 = {var_3: var_5}



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serialize_checked_type_with_no_serializer. Retrieved 2/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'some_format'



# Parsed testcases at query #45
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = (var_1, var_2)
    var_6 = lambda _: var_5
    var_7 = [var_4, var_6]
    var_8 = module_1.check_global_invariants(var_0, var_7)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda _: var_7
    var_9 = [var_4, var_8]
    var_10 = module_1.check_global_invariants(var_0, var_9)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 'ERROR1'
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = 'ERROR2'
    var_6 = (var_1, var_5)
    var_7 = lambda _: var_6
    var_8 = [var_4, var_7]
    var_9 = module_1.check_global_invariants(var_0, var_8)

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = []
    var_2 = module_1.check_global_invariants(var_0, var_1)



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 15/22 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'MockItemType'
    var_4 = ()
    var_5 = {}
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = 'MockType'
    var_11 = ()
    var_12 = 'create'
    var_13 = lambda self, data, _factory_fields=None: data
    var_14 = {var_12: var_13}



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 2/4 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'some_format'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 17/22 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 1
    var_8 = [var_7]
    var_9 = None
    var_10 = True
    var_11 = lambda : var_10
    var_12 = True
    var_13 = lambda : var_12
    var_14 = True
    var_15 = lambda : var_14
    var_16 = {var_2: var_8, var_3: var_9, var_4: var_11, var_5: var_13, var_6: var_15}



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_sequence_field_optional_predicate. Retrieved 2/3 statements.


def test_case_0():
    var_0 = True
    var_1 = []



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_restore_seq_field_pickle_returns_correct_type. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 'MockCheckedClass'
    var_1 = ()
    var_2 = {}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = 'MockType'
    var_8 = ()
    var_9 = 'create'
    var_10 = lambda self, data, **kwargs: data
    var_11 = {var_9: var_10}



# Parsed testcases at query #51
#--------------------------




import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = lambda _: var_3
    var_5 = [var_4]
    var_6 = module_1.check_global_invariants(var_0, var_5)



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pfield_constructor_with_all_parameters. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)



# Parsed testcases at query #53
#--------------------------

# Failed to parse test_pmap_field_creates_checked_pmap_field.
# Partially parsed test_pmap_field_with_optional_true. Retrieved 2/12 statements.
# Partially parsed test_pmap_field_with_invariant. Retrieved 1/2 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    var_0 = True

def test_case_0():
    var_0 = True



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_check_field_parameters_with_valid_field. Retrieved 4/8 statements.
# Partially parsed test_check_field_parameters_with_invalid_type_parameter. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_invalid_initial_type. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_invariant. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_factory. Retrieved 4/9 statements.
# Partially parsed test_check_field_parameters_with_non_callable_serializer. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 0
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 3.14
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 0
    var_1 = 'not_callable'
    var_2 = lambda : 0
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 0
    var_1 = lambda x: True
    var_2 = 'not_callable'
    var_3 = lambda x: str(x)

def test_case_0():
    var_0 = 0
    var_1 = lambda x: True
    var_2 = lambda : 0
    var_3 = 'not_callable'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_optional_type_when_optional_is_true. Retrieved 1/3 statements.


def test_case_0():
    var_0 = True



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_serialize_with_checked_type_and_no_serializer. Retrieved 3/5 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/5 statements.


import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = module_0.CheckedType()
    var_1 = 'serialized'
    var_2 = 'format'

def test_case_0():
    var_0 = 'json'
    var_1 = 'data'



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_restore_pmap_field_pickle. Retrieved 12/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = (var_0, var_1)
    var_3 = 2
    var_4 = 'b'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'MockPMapField'
    var_8 = ()
    var_9 = 'create'
    var_10 = lambda data, _factory_fields: data
    var_11 = {var_9: var_10}



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pfield_initialization. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = lambda x: x > var_0
    var_2 = 0
    var_3 = True
    var_4 = lambda : var_0
    var_5 = lambda x: str(x)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_restore_seq_field_pickle. Retrieved 8/15 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = {}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]
    var_7 = set()



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 16/22 statements.


def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'initial'
    var_4 = 'invariant'
    var_5 = 'factory'
    var_6 = 'serializer'
    var_7 = 123
    var_8 = [var_7]
    var_9 = 'initial_value'
    var_10 = True
    var_11 = lambda x: var_10
    var_12 = None
    var_13 = lambda : var_12
    var_14 = lambda x: x
    var_15 = {var_2: var_8, var_3: var_9, var_4: var_11, var_5: var_13, var_6: var_14}



