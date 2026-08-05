####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pclassmetameta_new_basic. Retrieved 5/19 statements.
# Partially parsed test_pclassmetameta_new_with_inheritance. Retrieved 1/6 statements.
# Failed to parse test_pclassmetameta_new_is_pclass_logic.


def test_case_0():
    var_0 = 'some_field'
    var_1 = 'NewClass'
    var_2 = '_pclass_fields'
    var_3 = '_pclash_invariants'
    var_4 = '_pclass_invariants'
    var_5 = '_pclass_frozen'

def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_frozen'



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_is_pclass_with_checkedtype.
# Failed to parse test_is_pclass_with_multiple_bases_including_checkedtype.
# Failed to parse test_is_pclass_with_single_different_base.


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



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_constructor_success. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_with_all_fields. Retrieved 3/4 statements.
# Partially parsed test_pclass_constructor_raises_attribute_error_on_extra_fields. Retrieved 2/4 statements.
# Failed to parse test_pclass_constructor_raises_invariant_exception_on_missing_mandatory_field.
# Partially parsed test_pclass_constructor_immutability_on_init. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = 'z'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = 'are not among the specified fields'

def test_case_0():
    var_0 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclassmetamethod_executes_successfully. Retrieved 7/19 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = '_pclass_fields'
    var_2 = '__invariant__'
    var_3 = {}
    var_4 = True
    var_5 = lambda x: (var_4, x)
    var_6 = {var_1: var_3, var_2: var_5}
    var_7 = '_pclass_fields'
    var_8 = bool('_pclass_fields' in var_6)
    assert var_8 is True
    var_9 = '_pclass_invariants'
    var_10 = bool('_pclass_invariants' in var_6)
    assert var_10 is True
    var_11 = '__slots__'
    var_12 = bool('__slots__' in var_6)
    assert var_12 is True
    var_13 = '__weakref__'
    var_14 = bool('__weakref__' in var_6['__slots__'])
    assert var_14 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_PClassMeta__new_basic_functionality. Retrieved 6/22 statements.
# Failed to parse test_PClassMeta__new_with_checked_type_adds_weakref.


def test_case_0():
    var_0 = 'some_field'
    var_1 = 'f1'
    var_2 = 'TestClass'
    var_3 = 'inherited_field'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = lambda x: True
    var_7 = 'new_value'
    var_8 = '_pclass_fields'
    var_9 = 'new_field'
    var_10 = 'inherited_field'
    var_11 = '_pclass_invariants'
    var_12 = '__slots__'
    var_13 = '_pclass_frozen'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_constructor_valid_args. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_multiple_args. Retrieved 3/4 statements.
# Partially parsed test_pclass_constructor_invalid_extra_field. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 1/3 statements.
# Partially parsed test_pclass_constructor_immutability_on_setattr. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_deletion_protection. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 5
    var_1 = 'z'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = True
    var_2 = 'are not among the specified fields'

def test_case_0():
    var_0 = 5
    var_1 = 'PClass.x'

def test_case_0():
    var_0 = 1
    var_1 = "Can't set attribute"

def test_case_0():
    var_0 = 1
    var_1 = "Can't delete attribute"



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_set_with_kwargs. Retrieved 3/5 statements.
# Partially parsed test_pclass_set_with_positional_args. Retrieved 4/6 statements.
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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_constructor_valid_args. Retrieved 1/2 statements.
# Partially parsed test_pclass_constructor_all_fields. Retrieved 3/4 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_immutability_on_init. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_deletion_fails. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_factory_fields_logic. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 5
    var_1 = "Should have raised InvariantException for missing mandatory field 'x'"
    var_2 = AssertionError(var_1)
    var_3 = 'TestPClass.x'

def test_case_0():
    var_0 = 1
    var_1 = 'not_allowed'
    var_2 = 'Should have raised AttributeError for extra fields'
    var_3 = AssertionError(var_2)
    var_4 = 'extra'

def test_case_0():
    var_0 = 1
    var_1 = 'Should not be able to set attribute on frozen PClass'
    var_2 = AssertionError(var_1)
    var_3 = "Can't set attribute"

def test_case_0():
    var_0 = 1
    var_1 = 'Should not be able to delete attributes on PClass'
    var_2 = AssertionError(var_1)
    var_3 = "Can't delete attribute"

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = {var_0}



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_test_new_raises_invariant_exception_on_missing_mandatory_field.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_eq_equality. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_inequality. Retrieved 5/9 statements.
# Partially parsed test_pclass_eq_different_class. Retrieved 3/8 statements.


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



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_repr_with_fields. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_with_different_types. Retrieved 5/9 statements.


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
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_persistent_returns_original_when_not_dirty. Retrieved 3/9 statements.
# Partially parsed test_persistent_returns_new_instance_when_dirty. Retrieved 5/13 statements.
# Partially parsed test_persistent_updates_dirty_flag_after_removal. Retrieved 5/13 statements.
# Partially parsed test_persistent_set_same_value_does_not_mark_dirty. Retrieved 3/10 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = 'b'
    var_4 = 2
    var_5 = 'b'

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'a'

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclassmetamethods_new_basic. Retrieved 13/34 statements.
# Partially parsed test_pclassmetamethods_new_with_pfields. Retrieved 2/14 statements.
# Partially parsed test_pclassmetamethods_new_is_pclass. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = ()
    var_2 = 'TestClass'
    var_3 = '__invariant__'
    var_4 = True
    var_5 = ()
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_6
    var_8 = {var_3: var_7}
    var_9 = 'NewClass'
    var_10 = '_pclass_fields'
    var_11 = '_pclass_invariants'
    var_12 = '__slots__'
    var_13 = '_pclass_frozen'

def test_case_0():
    var_0 = 'a'
    var_1 = 'Child'
    var_2 = '_pclass_fields'
    var_3 = 'a'
    var_4 = 'a'

def test_case_0():
    var_0 = {}
    var_1 = 'CheckedClass'
    var_2 = '__weakref__'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_reduce_returns_correct_reduction_tuple. Retrieved 3/7 statements.
# Partially parsed test_pclass_reduce_handles_only_present_fields. Retrieved 2/4 statements.
# Partially parsed test_pclass_reduce_equality_of_data. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = 0

def test_case_0():
    var_0 = 5
    var_1 = None
    var_2 = 'x'
    var_3 = 'y'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = {var_2: var_0, var_3: var_1}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclassmetane_new_basic. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 1
    var_1 = 10
    var_2 = lambda : (True, [])
    var_3 = '_pclass_fields'
    var_4 = '_pclass_invariants'
    var_5 = 'some_field'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_remove_non_existent_raises_error. Retrieved 9/12 statements.


import builtins as module_0
import pyrsistent._pclass as module_1

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
    var_12 = module_1._PClassEvolver(var_6, var_11)
    var_13 = var_12.remove(var_7)
    var_14 = 'a'
    var_15 = bool('a' not in var_12._pclass_evolver_data)
    assert var_15 is True
    var_16 = var_12._pclass_evolver_data['b']
    assert var_16 == 2
    var_17 = var_12._pclass_evolver_data_is_dirty
    assert var_17 is True

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
import pyrsistent._pclass as module_1

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
    var_10 = module_1._PClassEvolver(var_6, var_9)
    var_11 = var_10.set(var_7, var_8)
    var_12 = 'a'
    var_13 = bool('a' in var_10._factory_fields)
    assert var_13 is True
    var_14 = var_10.remove(var_7)
    var_15 = 'a'
    var_16 = bool('a' not in var_10._factory_fields)
    assert var_16 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_eq_with_same_class_instance. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_constructor_valid_args. Retrieved 1/2 statements.
# Partially parsed test_pclass_constructor_all_args. Retrieved 3/4 statements.
# Partially parsed test_pclass_constructor_raises_attribute_error_on_extra_fields. Retrieved 2/4 statements.
# Failed to parse test_pclass_constructor_raises_invariant_exception_on_missing_mandatory_fields.
# Partially parsed test_pclass_constructor_immutability_setattr. Retrieved 1/4 statements.
# Partially parsed test_pclass_constructor_delattr_raises_error. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 5
    var_1 = 20
    var_2 = 30

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 'are not among the specified fields'

def test_case_0():
    var_0 = 5
    var_1 = "Can't set attribute"

def test_case_0():
    var_0 = 5
    var_1 = "Can't delete attribute"



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_pclass_fields_not_empty.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serialize_basic. Retrieved 2/4 statements.
# Partially parsed test_serialize_equality. Retrieved 2/6 statements.
# Partially parsed test_serialize_different_values. Retrieved 3/7 statements.
# Partially parsed test_serialize_representation. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 5
    var_1 = True
    var_2 = 'x=5'
    var_3 = 'y=True'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_constructor_basic_instantiation. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_raises_error_on_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_raises_error_on_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_immutability. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_equality. Retrieved 5/10 statements.
# Partially parsed test_pclass_constructor_hashable. Retrieved 2/8 statements.
# Partially parsed test_pclass_constructor_repr. Retrieved 4/8 statements.
# Partially parsed test_pclass_constructor_deletion_error. Retrieved 2/8 statements.


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
    var_2 = 'MandatoryClass.x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'y'

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
    var_4 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'a'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = "Can't delete attribute"



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclassmetamethods_new_basic. Retrieved 3/19 statements.
# Partially parsed test_pclassmetamethods_new_inheritance_and_slots. Retrieved 2/12 statements.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = 'x'
    var_2 = '_pclass_invariants'
    var_3 = 0
    var_4 = '__slots__'
    var_5 = '_pclass_frozen'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = '_pclass_frozen'

def test_case_0():
    var_0 = '__weakref__'

def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pclass_reduce_returns_correct_tuple. Retrieved 2/4 statements.
# Partially parsed test_pclass_pickling_works. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'test'

def test_case_0():
    var_0 = 42
    var_1 = 'hello'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/8 statements.
# Partially parsed test_pcall_new_extra_attribute_error. Retrieved 2/9 statements.
# Partially parsed test_pclass_new_field_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_factory_fields_logic. Retrieved 4/9 statements.
# Failed to parse test_pclass_new_initial_callable.
# Partially parsed test_pclass_new_ignore_extra_param. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'A.x'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = bool(var_0)
    assert var_2 is True
    var_3 = "y' are not among the specified fields"
    var_4 = bool("y' are not among the specified fields" in var_1)
    assert var_4 is True

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 'Invalid type for field B.x'
    var_2 = bool('Invalid type for field B.x' in var_0)
    assert var_2 is True

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 'x'
    var_3 = {var_2}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = bool(var_0)
    assert var_3 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pclassmetanew_basic. Retrieved 1/15 statements.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = 'field1'
    var_2 = '_pclass_invariants'
    var_3 = '__slots__'
    var_4 = '_pclass_frozen'
    var_5 = 'field1'

def test_case_0():
    var_0 = '__weakref__'

def test_case_0():
    var_0 = 'value'
    var_1 = 'new_value'
    var_2 = 'base_field'
    var_3 = 'derived_field'
    var_4 = '_pclass_invariants'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_reduce_pickling_support. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'test'



# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serialize_basic_functionality. Retrieved 2/4 statements.
# Partially parsed test_serialize_equality_check. Retrieved 2/6 statements.
# Partially parsed test_serialize_value_difference. Retrieved 3/7 statements.
# Partially parsed test_serialize_uniqueness_of_fields. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 'z'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_pclass_repr_format. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'hello'
    var_2 = "TestPClassRepr(x=1, y='hello')"



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 2/3 statements.
# Partially parsed test_pclass_new_type_error. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_extra_fields_error. Retrieved 6/12 statements.
# Partially parsed test_pclass_new_equality. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_hash. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_setattr_frozen. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'val'

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = 'val'
    var_2 = bool(var_1)
    assert var_2 is True

def test_case_0():
    var_0 = 10
    var_1 = bool(var_0)
    assert var_1 is True
    var_2 = 'TestPClass.z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestAClass'
    var_1 = 'a'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 2
    var_6 = bool(var_0)
    assert var_6 is True
    var_7 = 'unknown'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = 2

def test_case_0():
    var_0 = 1
    var_1 = 'a'

def test_case_0():
    var_0 = 1
    var_1 = 'a'



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_new_handles_empty_fields.




# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_inequality. Retrieved 6/15 statements.
# Partially parsed test_pclass_hash_with_different_types. Retrieved 4/11 statements.
# Partially parsed test_pclass_hash_consistency. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'
    var_4 = 2
    var_5 = 'different'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'not a pclass'
    var_3 = hash(var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_pclass_reduce_serialization. Retrieved 2/5 statements.
# Partially parsed test_pclass_reduce_handles_missing_fields. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'

def test_case_0():
    var_0 = 5
    var_1 = None



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_set_args_expansion. Retrieved 4/6 statements.
# Partially parsed test_set_kwargs_expansion. Retrieved 3/5 statements.
# Partially parsed test_set_factory_fields_logic. Retrieved 3/5 statements.
# Partially parsed test_set_args_and_kwargs_together. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 3

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'y'
    var_3 = 5
    var_4 = 10



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pclass_repr_format. Retrieved 3/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = "TestClass(x=10, y='hello')"



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pclass_repr_basic. Retrieved 4/8 statements.
# Partially parsed test_pclass_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_repr_different_types. Retrieved 8/12 statements.
# Partially parsed test_pclass_repr_order_consistency. Retrieved 4/10 statements.


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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = [var_2, var_3]
    var_5 = 'key'
    var_6 = 'val'
    var_7 = {var_5: var_6}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pclass_constructor_success. Retrieved 5/11 statements.
# Partially parsed test_pclass_constructor_raises_attribute_error_for_extra_fields. Retrieved 5/10 statements.
# Partially parsed test_pclass_constructor_raises_invariant_exception_for_missing_mandatory_fields. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_factory_fields_logic. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_immutability. Retrieved 4/10 statements.
# Partially parsed test_pclass_constructor_deletion_fails. Retrieved 4/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = False
    var_4 = module_0.field(mandatory=var_3)
    var_5 = 5
    var_6 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'are not among the specified fields for TestClass'
    var_4 = 'Should have raised AttributeError'
    var_5 = AssertionError(var_4)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'TestClass.x'
    var_3 = 'Should have raised InvariantException'
    var_4 = AssertionError(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = {var_2}
    var_4 = 5
    var_5 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = "Can't set attribute"
    var_3 = 'Should not be able to set attribute on frozen PClass'
    var_4 = AssertionError(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'use remove()'
    var_3 = 'Should not be able to delete attribute'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_invalid_type. Retrieved 2/9 statements.
# Partially parsed test_pclass_new_missing_mandatory. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_extra_kwargs_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_field_invariant_failure. Retrieved 3/10 statements.
# Partially parsed test_pclass_new_with_factory_fields_exclusion. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_initial_callable. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = lambda x: x
    var_3 = 'default'
    var_4 = 10

def test_case_0():
    var_0 = 'x'
    var_1 = 'not_an_int'
    var_2 = 'Invalid type'

def test_case_0():
    var_0 = 'x'
    var_1 = True
    var_2 = 'TestPClassNew.x'

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = 'y'

def test_case_0():
    var_0 = 'x'
    var_1 = lambda x: x
    var_2 = 1
    var_3 = 'error_code_123'

def test_case_0():
    var_0 = 'x'
    var_1 = 10
    var_2 = {var_0}

def test_case_0():
    var_0 = 'x'
    var_1 = 5
    var_2 = lambda : var_1



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_pclass_new_with_no_fields.




# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclassmetanew_basic_functionality. Retrieved 1/16 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = '_pclass_fields'
    var_4 = 'field_a'
    var_5 = '_pclass_frozen'
    var_6 = '_pclass_invariants'

def test_case_0():
    var_0 = '_pclass_frozen'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = '_pclass_frozen'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_missing_mandatory_field. Retrieved 1/6 statements.
# Partially parsed test_pquanto_new_extra_fields_error. Retrieved 3/8 statements.
# Failed to parse test_pclass_new_initial_callable.
# Partially parsed test_pclass_new_with_factory_fields_filtering. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_invariant_failure. Retrieved 2/11 statements.
# Failed to parse test_pclass_new_field_invariant_failure.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 'MandatoryClass.x'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = str(var_0)
    var_3 = 'z'
    var_4 = bool('z' in var_2)
    assert var_4 is True

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = str(var_0)
    var_2 = 'ERR01'
    var_3 = bool('ERR01' in var_1)
    assert var_3 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pclass_new_success. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_raises_attribute_error_on_extra_kwargs. Retrieved 5/10 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_on_missing_mandatory. Retrieved 4/9 statements.
# Partially parsed test_pclass_new_raises_invariant_exception_on_field_invariant_failure. Retrieved 5/12 statements.
# Partially parsed test_pclass_new_with_factory_fields_filtering. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra_logic. Retrieved 4/11 statements.


def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = lambda v: v.upper()
    var_4 = 'hello'

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 2
    var_3 = "'extra' are not among the specified fields for TestClass"
    var_4 = 'Should have raised AttributeError'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'x'
    var_1 = True
    var_2 = 'TestClass.x'
    var_3 = 'Should have raised InvariantException for missing field'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 'x'
    var_1 = lambda v: v
    var_2 = 1
    var_3 = 'ERR_CODE'
    var_4 = 'Should have raised InvariantException for field invariant failure'
    var_5 = AssertionError(var_4)

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 1
    var_4 = 2
    var_5 = "'y' are not among the specified fields for TestClass"

def test_case_0():
    var_0 = 'x'
    var_1 = 1
    var_2 = 'not_a_field'
    var_3 = True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_reduce_returns_correct_structure. Retrieved 3/7 statements.
# Partially parsed test_pclass_reduce_handles_partial_attributes. Retrieved 4/7 statements.
# Partially parsed test_pclass_reduce_equality_of_data. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = 0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 6
    var_4 = 'a'
    var_5 = 'b'

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_pclass_fields_not_empty.




# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclassmetaclass_new_does_not_add_weakref_when_no_pclass_bases. Retrieved 7/17 statements.


def test_case_0():
    var_0 = '__invariant__'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'
    var_4 = True
    var_5 = lambda x: var_4
    var_6 = {var_0: var_5}
    var_7 = '__weakref__'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_constructor_success. Retrieved 1/2 statements.
# Partially parsed test_pclass_constructor_all_fields. Retrieved 3/4 statements.
# Partially parsed test_pclass_constructor_extra_fields_raises_error. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_missing_mandatory_fields_raises_error. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_immutability. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_deletion_fails. Retrieved 3/8 statements.
# Partially parsed test_pclass_constructor_equality. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 5
    var_1 = 10
    var_2 = 'are not among the specified fields'
    var_3 = 'Should have raised AttributeError for extra field'
    var_4 = AssertionError(var_3)

def test_case_0():
    var_0 = 10
    var_1 = 'TestClass.x'
    var_2 = 'Should have raised exception for missing mandatory field'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 5
    var_1 = 'PClass should be frozen and not allow attribute assignment'
    var_2 = AssertionError(var_1)

def test_case_0():
    var_0 = 5
    var_1 = "Can't delete attribute"
    var_2 = 'PClass should not allow deleting attributes'
    var_3 = AssertionError(var_2)

def test_case_0():
    var_0 = 5
    var_1 = 6



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_set_with_kwargs. Retrieved 3/5 statements.
# Partially parsed test_set_with_positional_args. Retrieved 4/6 statements.
# Partially parsed test_set_preserves_unmentioned_fields. Retrieved 4/7 statements.
# Partially parsed test_set_returns_new_instance_of_same_class. Retrieved 3/6 statements.


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
    var_3 = 'y'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_new_with_fields. Retrieved 2/7 statements.
# Failed to parse test_pclass_new_with_initial_values.
# Failed to parse test_pclass_new_with_mandatory_field_error.
# Partially parsed test_pclass_new_with_extra_kwargs_error. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_fields_subset. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'unknown'

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_check_and_set_attr_success. Retrieved 4/13 statements.
# Partially parsed test_check_and_set_attr_type_error. Retrieved 4/14 statements.
# Partially parsed test_check_and_set_attr_invariant_failure. Retrieved 4/16 statements.
# Partially parsed test_check_and_set_attr_multiple_types_success. Retrieved 4/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'age'
    var_2 = 25
    var_3 = len(var_0)
    assert var_3 == 0

def test_case_0():
    var_0 = []
    var_1 = 'age'
    var_2 = 'not_an_int'
    var_3 = 'Invalid type for field'
    var_4 = len(var_0)
    assert var_4 == 0

def test_case_0():
    var_0 = []
    var_1 = 'age'
    var_2 = -5
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = var_0[0]
    assert var_4 == 'error_negative'

def test_case_0():
    var_0 = []
    var_1 = 'data'
    var_2 = 'hello'
    var_3 = len(var_0)
    assert var_3 == 0



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_reduce_returns_correct_tuple. Retrieved 2/6 statements.
# Partially parsed test_pclass_pickling_works. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 10
    var_1 = 20

def test_case_0():
    var_0 = 10
    var_1 = 20



# Parsed testcases at query #15
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_basic_functionality. Retrieved 2/4 statements.
# Partially parsed test_serialize_equality. Retrieved 2/6 statements.
# Partially parsed test_serialize_different_values. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_constructor_valid_args. Retrieved 1/2 statements.
# Partially parsed test_pclass_constructor_all_fields. Retrieved 3/4 statements.
# Failed to parse test_pclass_constructor_missing_mandatory_raises_error.
# Partially parsed test_pclass_constructor_extra_args_raises_error. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_immutability_on_init. Retrieved 1/5 statements.
# Partially parsed test_pclass_constructor_equality. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_hashable. Retrieved 1/3 statements.
# Partially parsed test_pclass_constructor_repr. Retrieved 2/4 statements.
# Partially parsed test_pclass_constructor_deletion_error. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3

def test_case_0():
    var_0 = 1
    var_1 = 10

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1



# Parsed testcases at query #18
#--------------------------

# Failed to parse test_is_pclass_true_with_single_checkedtype.
# Failed to parse test_is_pclass_false_with_multiple_bases_including_checkedtype.
# Failed to parse test_is_pclass_false_with_single_different_type.


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



# Parsed testcases at query #19
#--------------------------

# Partially parsed test___repr__. Retrieved 5/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'abc'
    var_4 = 10



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



