####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_set_updates_value_and_marks_dirty. Retrieved 5/11 statements.
# Partially parsed test_set_does_nothing_if_value_is_identical. Retrieved 4/10 statements.
# Partially parsed test_set_returns_self_for_chaining. Retrieved 4/10 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 2
    var_7 = 'a'

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'a'

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = {}
    var_4 = 'b'
    var_5 = 3



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_remove_success. Retrieved 6/12 statements.
# Partially parsed test_remove_raises_attribute_error_on_missing_key. Retrieved 5/13 statements.
# Partially parsed test_remove_returns_self. Retrieved 4/10 statements.


import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 1
    var_4 = 'a'
    var_5 = 'b'
    var_6 = 2
    var_7 = {var_4: var_3, var_5: var_6}
    var_8 = 'a'
    var_9 = 'a'

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = 'non_existent_key'

import builtins as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_pclassmetameta_new_with_fields.
# Partially parsed test_pclassmetameta_is_pclass_logic. Retrieved 4/29 statements.


def test_case_0():
    var_0 = '_pclass_fields'

def test_case_0():
    var_0 = 'PClassSimulated'
    var_1 = '_pclass_fields'
    var_2 = {}
    var_3 = {var_1: var_2}
    var_4 = '__weakref__'
    var_5 = '__weakref__'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_PClassMeta_new_basic_functionality. Retrieved 5/15 statements.
# Partially parsed test_PClassMeta_new_with_inheritance. Retrieved 4/8 statements.


def test_case_0():
    var_0 = {}
    var_1 = lambda x: (True, ())
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'
    var_5 = '_pclass_frozen'

def test_case_0():
    var_0 = lambda x: (True, ())
    var_1 = 1
    var_2 = lambda x: (True, (1,))
    var_3 = 2
    var_4 = '_pclass_frozen'

def test_case_0():
    var_0 = '__slots__'



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Failed to parse test_pclass_new_with_no_fields.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 2/6 statements.
# Partially parsed test_pclass_hash_inequality. Retrieved 3/7 statements.
# Partially parsed test_pclass_hash_different_types. Retrieved 3/7 statements.
# Partially parsed test_pclass_hash_consistency. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '1'

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_raises_attribute_error_for_extra_kwargs. Retrieved 3/8 statements.
# Failed to parse test_pclass_new_raises_invariant_exception_for_missing_mandatory_fields.
# Partially parsed test_pclass_new_raises_invariant_exception_for_field_invariant_failure. Retrieved 1/15 statements.
# Partially parsed test_pclass_new_respects_factory_fields_filtering. Retrieved 4/15 statements.
# Failed to parse test_pclass_new_handles_initial_callable.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10
    var_1 = 20
    var_2 = str(var_0)
    var_3 = 'are not among the specified fields for TestClass'
    var_4 = bool('are not among the specified fields for TestClass' in var_2)
    assert var_4 is True

def test_case_0():
    var_0 = -5
    var_1 = 'negative_error'

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = 10
    var_3 = 20



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_inequality. Retrieved 5/11 statements.
# Partially parsed test_pclass_hash_different_types. Retrieved 3/7 statements.
# Partially parsed test_pclass_hash_with_none_values. Retrieved 4/10 statements.


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
    var_2 = hash(var_1)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = None



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_new_kwargs_not_empty. Retrieved 1/2 statements.


def test_case_0():
    var_0 = 10



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_remove_item_exists. Retrieved 5/11 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'a'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_eq_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_different_types. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_ne_operator. Retrieved 3/8 statements.


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
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pclass_eq_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_different_class. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_not_implemented. Retrieved 3/7 statements.


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
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 5



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_set_keyword_arguments. Retrieved 3/5 statements.
# Partially parsed test_set_positional_arguments. Retrieved 4/6 statements.
# Partially parsed test_set_immutability. Retrieved 3/5 statements.
# Partially parsed test_set_multiple_fields_at_once. Retrieved 4/6 statements.
# Partially parsed test_set_preserves_unspecified_fields. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'y'
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5
    var_3 = 'y'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test___reduce__. Retrieved 4/8 statements.
# Partially parsed test___reduce___with_partial_attributes. Retrieved 4/8 statements.


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
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test___repr__. Retrieved 2/4 statements.
# Partially parsed test___repr___with_different_order. Retrieved 2/4 statements.
# Partially parsed test___repr___with_missing_fields. Retrieved 5/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'abc'

def test_case_0():
    var_0 = 'abc'
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = module_0.field(initial=var_2)
    var_4 = None
    var_5 = module_0.field(initial=var_4)
    var_6 = 10
    var_7 = 'x=10'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_new_with_fields. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 5



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pclassmetaclass_new_not_pclass_bases. Retrieved 2/12 statements.


def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = '__weakref__'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_hash_functionality. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_on_missing_mandatory_field. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'MandatoryClass.x'



# Parsed testcases at query #21
#--------------------------

# Failed to parse test_pclass_missing_mandatory_field_raises_invariant_exception.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 2/3 statements.
# Partially parsed test_pclass_new_type_error. Retrieved 2/4 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/3 statements.
# Partially parsed test_pclass_new_extra_attribute_error. Retrieved 3/5 statements.
# Partially parsed test_pclass_new_initial_callable. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 5
    var_1 = 20

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 20
    var_2 = 'Invalid type for field TestClass.x'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 5
    var_1 = 'TestClass.z'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = 20
    var_2 = 100
    var_3 = "unknown' are not among the specified fields"
    var_4 = bool(False)
    assert var_4 is True

def test_case_0():
    pass

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_PClassMeta__new_basic_functionality. Retrieved 1/16 statements.


def test_case_0():
    var_0 = '__slots__'
    var_1 = '_pclass_frozen'
    var_2 = 'field1'

def test_case_0():
    var_0 = '__weakref__'
    var_1 = '_pclass_frozen'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '_pclass_frozen'



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_is_pclass_with_checked_type.
# Failed to parse test_is_pclass_with_multiple_bases_including_checked_type.
# Failed to parse test_is_pclass_with_different_single_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False

import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = None
    var_1 = (var_0,)
    var_2 = module_0._is_pclass(var_1)
    assert var_2 is False



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_set_kwargs. Retrieved 3/5 statements.
# Partially parsed test_pclass_set_positional_args. Retrieved 4/6 statements.
# Partially parsed test_pclass_set_multiple_fields_mixed. Retrieved 4/6 statements.
# Partially parsed test_pclass_set_preserves_unspecified_fields. Retrieved 3/5 statements.
# Partially parsed test_pclass_set_equality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_set_args_logic. Retrieved 4/6 statements.
# Partially parsed test_set_kwargs_logic. Retrieved 3/5 statements.
# Partially parsed test_set_with_args_and_kwargs_logic. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 10
    var_4 = 20



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_constructor_success. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_ignore_extra_parameter. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_factory_fields_restriction. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'TestClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = "y' are not among the specified fields for TestClass"

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pclass_eq_isinstance_true. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_remove_success. Retrieved 10/13 statements.
# Partially parsed test_remove_non_existent_raises_error. Retrieved 9/13 statements.
# Partially parsed test_remove_updates_factory_fields. Retrieved 10/14 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 1
    var_10 = 2
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = 'a'
    var_13 = 'a'

import builtins as module_0

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = 'non_existent'

import builtins as module_0

def test_case_0():
    var_0 = 'Mock'
    var_1 = ()
    var_2 = {}
    var_3 = [var_0, var_1, var_2]
    var_4 = {}
    var_5 = module_0.type(*var_3, **var_4)
    var_6 = var_5()
    var_7 = 'a'
    var_8 = 1
    var_9 = {var_7: var_8}
    var_10 = 'b'
    var_11 = 2
    var_12 = 'b'
    var_13 = 'b'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_repr_basic_functionality. Retrieved 4/8 statements.
# Partially parsed test_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_repr_with_none_value. Retrieved 2/6 statements.
# Partially parsed test_repr_equality_with_other_types. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pclass_constructor_basic_initialization. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_raises_error_on_missing_mandatory_field. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_raises_error_on_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields_restriction. Retrieved 6/8 statements.
# Partially parsed test_pclass_constructor_immutability_via_setattr. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_deletion_restriction. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
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
    var_4 = 'y'

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
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_is_pclass_true_with_single_checkedtype.
# Failed to parse test_is_pclass_false_with_multiple_bases.
# Failed to parse test_is_pclass_false_with_different_single_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 2/6 statements.
# Partially parsed test_pclass_hash_inequality. Retrieved 3/7 statements.
# Partially parsed test_pclass_hash_different_types. Retrieved 3/7 statements.
# Partially parsed test_pclass_hash_uniqueness_with_set. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '1'

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pclassmetamethod_new_basic_inheritance. Retrieved 2/26 statements.
# Partially parsed test_pclassmetamethod_new_with_pfields_logic. Retrieved 2/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = '_pclass_fields'
    var_4 = 'a'
    var_5 = '_pclass_frozen'
    var_6 = 'a'

def test_case_0():
    var_0 = 'field_a'
    var_1 = '_pclass_fields'
    var_2 = '_pclass_fields'
    var_3 = 'field_a'
    var_4 = 'field_a'

def test_case_0():
    var_0 = '__weakref__'
    var_1 = '_pclass_frozen'
    var_2 = '__weakref__'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_repr_basic_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_complex_types. Retrieved 7/11 statements.
# Failed to parse test_pclass_repr_empty_fields.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = (var_3, var_4)
    var_6 = [var_1, var_2, var_5]



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclassmetamethod_executes_successfully. Retrieved 4/64 statements.


def test_case_0():
    var_0 = 'Sub'
    var_1 = globals()
    var_2 = var_0 in var_1
    var_3 = bool(var_2 or True)
    assert var_3 is True
    var_4 = '__slots__'
    var_5 = '__weakref__'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_reduce_returns_correct_structure. Retrieved 4/8 statements.
# Partially parsed test_pclass_reduce_only_includes_existing_attributes. Retrieved 4/8 statements.


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
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_constructor_basic_initialization. Retrieved 1/2 statements.
# Partially parsed test_pclass_constructor_multiple_fields. Retrieved 3/4 statements.
# Partially parsed test_pclass_constructor_raises_attribute_error_on_extra_fields. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 1/3 statements.
# Partially parsed test_pclass_constructor_immutability_via_setattr. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_equality. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_hashable. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_repr. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 99
    var_2 = 'are not among the specified fields'

def test_case_0():
    var_0 = 10
    var_1 = 'TestClass.x'

def test_case_0():
    var_0 = 1
    var_1 = bool(True)
    assert var_1 is True

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_pclass_constructor_success. Retrieved 1/2 statements.
# Partially parsed test_pclass_constructor_with_extra_args_raises_error. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field_raises_error. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_immutability_on_setattr. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_deletable_attribute_raises_error. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_equality. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_hashable. Retrieved 2/9 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5
    var_1 = 100
    var_2 = 'Should have raised AttributeError'
    var_3 = AssertionError(var_2)
    var_4 = 'are not among the specified fields'

def test_case_0():
    var_0 = 'Should have raised InvariantException'
    var_1 = AssertionError(var_0)
    var_2 = 'TestClass.x'

def test_case_0():
    var_0 = 5
    var_1 = 'Should have raised AttributeError due to frozen state'
    var_2 = AssertionError(var_1)
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = 5
    var_1 = 'Should have raised AttributeError'
    var_2 = AssertionError(var_1)
    var_3 = "Can't delete attribute"

def test_case_0():
    var_0 = 5
    var_1 = 10

def test_case_0():
    var_0 = 5
    var_1 = 10



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_missing_mandatory_field_raises_invariant_exception. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'MandatoryClass.required_field'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 1/2 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/3 statements.
# Failed to parse test_pclass_new_raises_missing_mandatory.
# Partially parsed test_pclass_new_raises_invalid_type. Retrieved 1/3 statements.
# Partially parsed test_pclass_new_raises_invariant_failure. Retrieved 1/3 statements.
# Partially parsed test_pclass_new_raises_extra_attribute. Retrieved 2/4 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 2/3 statements.
# Partially parsed test_pclass_new_with_factory_fields_restriction. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = 10
    var_1 = 'custom'

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = bool(True)
    assert var_1 is True
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = -5
    var_1 = 'ERR_POS'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'unknown'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 5
    var_1 = True

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_raises_invariant_exception_on_missing_mandatory_field. Retrieved 1/8 statements.
# Partially parsed test_pclass_raises_invariant_exception_on_field_invariant_failure. Retrieved 3/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'Field invariant failed'
    var_3 = 'MandatoryClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = str(var_1)
    var_3 = 'Field invariant failed'
    var_4 = bool('Field invariant failed' in var_2)
    assert var_4 is True
    var_5 = 'error_negative_x'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pclass_set_with_kwargs. Retrieved 3/5 statements.
# Partially parsed test_pclass_set_with_args. Retrieved 4/6 statements.
# Partially parsed test_pclass_set_multiple_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_set_immutability. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_eq_same_instance. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 3/7 statements.
# Partially parsed test_pclass_eq_different_types. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_not_a_pclass. Retrieved 2/5 statements.


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
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_constructor_valid_fields. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_fields_raises_error. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_immutability. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_deletion_raises_error. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_factory_fields_logic. Retrieved 6/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 10
    var_3 = 'are not among the specified fields'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'PClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'use remove()'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 10
    var_4 = 'x'
    var_5 = {var_4}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_reduce_returns_correct_tuple. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 2/8 statements.
# Partially parsed test_ppersistent_new_extra_kwargs_raises_error. Retrieved 2/8 statements.
# Failed to parse test_pclass_new_invariant_failure.
# Partially parsed test_pclass_new_with_factory_fields_allows_extra. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_type_error. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 1
    var_3 = 'TestClass.x'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'are not among the specified fields'
    var_3 = bool('are not among the specified fields' in var_0)
    assert var_3 is True

def test_case_0():
    var_0 = 10
    var_1 = 'x'
    var_2 = {var_1}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 'Invalid type'
    var_2 = bool('Invalid type' in var_0)
    assert var_2 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_set_with_positional_args_populates_kwargs. Retrieved 6/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 10



