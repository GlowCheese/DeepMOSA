####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'new_key'
    var_4 = 'new_value'
    var_5 = var_2.set(var_3, var_4)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'key'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = module_1._PClassEvolver(var_0, var_3)
    var_5 = 'new_value'
    var_6 = var_4.set(var_1, var_5)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_1._PClassEvolver(var_0, var_3)
    var_5 = var_4.set(var_1, var_2)
    var_6 = set()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = var_2.set(var_3, var_4)



# Parsed testcases at query #2
#--------------------------




import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = var_2.persistent()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = var_2.set(var_3, var_4)
    var_6 = var_2.persistent()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = var_2.set(var_3, var_4)
    var_6 = 'field2'
    var_7 = 'value2'
    var_8 = var_2.set(var_6, var_7)
    var_9 = var_2.persistent()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1._PClassEvolver(var_0, var_5)
    var_7 = var_6.remove(var_1)
    var_8 = var_6.persistent()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_multiple_updates. Retrieved 8/12 statements.
# Partially parsed test_set_with_mixed_arguments. Retrieved 7/11 statements.


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
    var_4 = 'x'
    var_5 = 10
    var_6 = 20



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_reduce_returns_correct_tuple. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_repr_returns_correct_format. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 2/9 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 6/14 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_type_check. Retrieved 2/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 2

def test_case_0():
    var_0 = 1
    var_1 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = -1
    var_5 = -2

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
    var_1 = 5
    var_2 = 2
    var_3 = 'x'
    var_4 = {var_3}

def test_case_0():
    var_0 = 1
    var_1 = 'not an int'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclassmeta_new_with_single_checkedtype_base. Retrieved 6/8 statements.
# Partially parsed test_pclassmeta_new_with_multiple_bases. Retrieved 4/10 statements.
# Partially parsed test_pclassmeta_new_with_invariant. Retrieved 9/4 statements.
# Partially parsed test_pclassmeta_new_with_inherited_fields. Retrieved 5/8 statements.
# Partially parsed test_pclassmeta_new_with_inherited_invariant. Retrieved 4/14 statements.
# Partially parsed test_pclassmeta_new_with_non_callable_invariant. Retrieved 4/7 statements.
# Partially parsed test_pclassmeta_new_with_no_checkedtype_base. Retrieved 4/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = module_0._PField()
    var_3 = module_0._PField()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'TestClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = module_0._PField()
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'x'
    var_5 = module_0._PField()
    var_6 = 'TestClass'
    var_7 = 0
    var_8 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'x'
    var_5 = module_0._PField()
    var_6 = 'TestClass'
    var_7 = 0
    var_8 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = 'y'
    var_2 = module_0._PField()
    var_3 = {var_1: var_2}
    var_4 = 'TestClass'

def test_case_0():
    var_0 = {}
    var_1 = 'TestClass'
    var_2 = 0
    var_3 = None

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not_callable'
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0._PField()
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invariant_error. Retrieved 3/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_frozen_attribute. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

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
    var_2 = -1
    var_3 = -2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.
# Partially parsed test_serialize_missing_field. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'test'
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'test'
    var_3 = 'str'
    var_4 = module_0.serialize(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = module_0.serialize()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/9 statements.
# Partially parsed test_pclass_hash_different_instances_same_values. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_instances_different_values. Retrieved 6/12 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 2/9 statements.
# Partially parsed test_pclass_hash_with_none_values. Retrieved 3/9 statements.


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
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'json'
    var_4 = module_0.serialize(var_3)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test__check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/13 statements.
# Partially parsed test__check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test__check_and_set_attr_with_failed_invariant. Retrieved 3/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 10

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 'not_an_int'

def test_case_0():
    var_0 = []
    var_1 = 'test_field'
    var_2 = 10



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_pickling_returns_restore_pickle_tuple. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_eq_same_instance. Retrieved 4/7 statements.
# Partially parsed test_eq_different_instances_same_values. Retrieved 4/8 statements.
# Partially parsed test_eq_different_values. Retrieved 6/10 statements.
# Partially parsed test_eq_different_classes. Retrieved 3/8 statements.
# Partially parsed test_eq_non_pclass_instance. Retrieved 2/5 statements.


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



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_repr_contains_class_name_and_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #18
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 15/22 statements.
# Failed to parse test_pclass_new_with_missing_mandatory_field.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Failed to parse test_pclass_new_with_initial_value.
# Failed to parse test_pclass_new_with_callable_initial.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/10 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_factory_field_and_ignore_extra. Retrieved 4/10 statements.
# Partially parsed test_pclass_new_with_factory_fields_param. Retrieved 5/11 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1
import pyrsistent._pset as module_2

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_2, var_3]
    var_5 = 'a'
    var_6 = {var_5: var_1}
    var_7 = 'b'
    var_8 = 'c'
    var_9 = {var_7, var_8}
    var_10 = [var_2, var_3]
    var_11 = {var_5: var_1}
    var_12 = module_1.pmap(var_11)
    var_13 = {var_7, var_8}
    var_14 = module_2.pset(var_13)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2

def test_case_0():
    var_0 = 'hello'

def test_case_0():
    var_0 = 'x'
    var_1 = 'hello'
    var_2 = {var_0: var_1}
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = 'x'
    var_4 = {var_3}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_fields_items_iteration. Retrieved 2/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 1/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_serialize_returns_dict_with_all_fields. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'new_key'
    var_4 = 'value'
    var_5 = var_2.set(var_3, var_4)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_1._PClassEvolver(var_0, var_3)
    var_5 = var_4.set(var_1, var_2)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'key'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = module_1._PClassEvolver(var_0, var_3)
    var_5 = 'new_value'
    var_6 = var_4.set(var_1, var_5)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = var_2.set(var_3, var_4)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 1/6 statements.
# Failed to parse test_pclass_constructor_with_missing_mandatory_field.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 6/11 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 4/9 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

def test_case_0():
    var_0 = 'x'
    var_1 = {var_0}
    var_2 = 1
    var_3 = 'value'

def test_case_0():
    var_0 = -1



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 3/10 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/11 statements.
# Partially parsed test_pclass_new_with_ignore_extra_and_factory. Retrieved 9/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'not an int'
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2

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
    var_1 = 5
    var_2 = 2
    var_3 = 'x'
    var_4 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 'z'
    var_4 = 5
    var_5 = 2
    var_6 = 3
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = True



# Parsed testcases at query #4
#--------------------------




import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1._PClassEvolver(var_0, var_5)
    var_7 = var_6.remove(var_1)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1._PClassEvolver(var_0, var_5)
    var_7 = 'c'
    var_8 = var_6.remove(var_7)



# Parsed testcases at query #5
#--------------------------

# Failed to parse test__is_pclass_with_single_checked_type_base.
# Failed to parse test__is_pclass_with_multiple_bases.
# Failed to parse test__is_pclass_with_non_checked_type_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_invalid_field_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

def test_case_0():
    var_0 = 'not an int'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

def test_case_0():
    var_0 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclassmeta_new_with_single_checkedtype_base. Retrieved 15/19 statements.
# Partially parsed test_pclassmeta_new_with_multiple_bases. Retrieved 12/18 statements.
# Partially parsed test_pclassmeta_new_with_inherited_invariants. Retrieved 12/19 statements.
# Partially parsed test_pclassmeta_new_with_no_invariant. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = '_pclass_fields'
    var_4 = '_pclass_invariants'
    var_5 = '__slots__'
    var_6 = True
    var_7 = lambda self: var_6
    var_8 = module_1.wrap_invariant(var_7)
    var_9 = (var_8,)
    var_10 = 'field1'
    var_11 = 'field2'
    var_12 = module_0.field()
    var_13 = module_0.field()
    var_14 = {var_10: var_12, var_11: var_13}

import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0.field()
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'
    var_5 = True
    var_6 = lambda self: var_5
    var_7 = module_1.wrap_invariant(var_6)
    var_8 = (var_7,)
    var_9 = 'field1'
    var_10 = module_0.field()
    var_11 = {var_9: var_10}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0.field()
    var_2 = lambda self: False
    var_3 = module_0.field()
    var_4 = '_pclass_fields'
    var_5 = '_pclass_invariants'
    var_6 = '__slots__'
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = module_0.field()
    var_10 = module_0.field()
    var_11 = {var_7: var_9, var_8: var_10}

def test_case_0():
    var_0 = 'not callable'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = '__slots__'
    var_4 = 'field1'
    var_5 = module_0.field()
    var_6 = {var_4: var_5}



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 8/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.field()
    var_5 = module_0.field()
    var_6 = 'hello'
    var_7 = 3.14



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'test'
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'
    var_4 = 'json'
    var_5 = module_0.serialize(var_4)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_extra_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_preserves_other_fields. Retrieved 7/11 statements.
# Partially parsed test_set_returns_new_instance. Retrieved 3/7 statements.
# Partially parsed test_set_with_multiple_updates. Retrieved 6/10 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10
    var_5 = 20



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_equality_with_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_equality_with_different_values. Retrieved 6/10 statements.
# Partially parsed test_pclass_equality_with_different_types. Retrieved 3/8 statements.
# Partially parsed test_pclass_equality_with_non_pclass_instance. Retrieved 2/5 statements.


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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields_raises_exception. Retrieved 3/11 statements.


def test_case_0():
    var_0 = -1
    var_1 = 1
    var_2 = 1



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/13 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 2/9 statements.
# Partially parsed test_pclass_hash_with_different_field_values. Retrieved 6/12 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_pickle_reduce_returns_correct_tuple. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_persistent_with_dirty_data. Retrieved 9/16 statements.
# Partially parsed test_persistent_with_clean_data. Retrieved 8/12 statements.
# Partially parsed test_persistent_after_removal. Retrieved 9/17 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 3

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = set()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #20
#--------------------------




import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = var_2.persistent()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'key'
    var_4 = 'value'
    var_5 = var_2.set(var_3, var_4)
    var_6 = var_2.persistent()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = module_1._PClassEvolver(var_0, var_1)
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = var_2.set(var_3, var_4)
    var_6 = 'field2'
    var_7 = 'value2'
    var_8 = var_2.set(var_6, var_7)
    var_9 = var_2.persistent()

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1._PClassEvolver(var_0, var_5)
    var_7 = var_6.remove(var_1)
    var_8 = var_6.persistent()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_failing_invariant. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_string_type. Retrieved 4/12 statements.
# Partially parsed test_check_and_set_attr_with_custom_type. Retrieved 2/16 statements.


def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 'not_an_int'

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = (var_0,)
    var_2 = []
    var_3 = 'attr'
    var_4 = 42

def test_case_0():
    var_0 = []
    var_1 = 'attr'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Failed to parse test_pclass_constructor_with_initial_values.
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
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_hash_returns_consistent_value. Retrieved 4/10 statements.
# Partially parsed test_hash_different_instances_different_values. Retrieved 6/12 statements.
# Partially parsed test_hash_with_missing_optional_field. Retrieved 2/9 statements.
# Partially parsed test_hash_with_initial_value. Retrieved 2/9 statements.


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
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2



# Parsed testcases at query #24
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_eq_with_same_instance. Retrieved 4/7 statements.
# Partially parsed test_eq_with_equivalent_instance. Retrieved 4/8 statements.
# Partially parsed test_eq_with_different_instance. Retrieved 6/10 statements.
# Partially parsed test_eq_with_different_class. Retrieved 3/8 statements.
# Partially parsed test_eq_with_non_pclass_instance. Retrieved 2/5 statements.
# Partially parsed test_eq_with_missing_fields. Retrieved 3/8 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #26
#--------------------------

# Failed to parse test_invariant_errors_or_missing_fields_predicate.




# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 3/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'type'



# Parsed testcases at query #28
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pclass_eq_with_same_instance. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_with_different_instance. Retrieved 6/10 statements.
# Partially parsed test_pclass_eq_with_different_class. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_non_pclass. Retrieved 2/5 statements.


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



# Parsed testcases at query #30
#--------------------------

# Failed to parse test_invariant_errors_or_missing_fields.




# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #32
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_repr_returns_correct_format. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pclass_pickling_support. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_fields. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 7/16 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 7/17 statements.
# Partially parsed test_check_and_set_attr_with_failed_invariant. Retrieved 11/21 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = []
    var_5 = 'test_field'
    var_6 = 'not_an_int'

def test_case_0():
    var_0 = 0
    var_1 = False
    var_2 = 'INVALID'
    var_3 = (var_1, var_2)
    var_4 = True
    var_5 = None
    var_6 = (var_4, var_5)
    var_7 = lambda x: var_3 if x < var_0 else var_6
    var_8 = []
    var_9 = 'test_field'
    var_10 = -1



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #39
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #40
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_equality_with_same_class_and_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_repr_format_matches_expected. Retrieved 5/12 statements.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = module_0.PClass()
    var_1 = '{0}({1})'
    var_2 = ', '
    var_3 = module_0.PClass()
    var_4 = '{0}={1}'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #45
#--------------------------




def test_case_0():
    var_0 = 'error1'
    var_1 = [var_0]
    var_2 = 'field1'
    var_3 = [var_2]



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 2/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = -1



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_invariant. Retrieved 17/21 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'Field'
    var_4 = ()
    var_5 = 'type'
    var_6 = 'invariant'
    var_7 = None
    var_8 = True
    var_9 = (var_8, var_7)
    var_10 = lambda x: var_9
    var_11 = {var_5: var_7, var_6: var_10}
    var_12 = 'test_field'
    var_13 = 'test_value'
    var_14 = module_0.object()
    var_15 = []
    var_16 = getattr(var_14, var_12)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #49
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #50
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #52
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serialize_includes_all_fields. Retrieved 5/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = -1
    var_2 = 1



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/13 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 1
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 1
    var_3 = 'json'
    var_4 = module_0.serialize(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 1
    var_4 = module_0.serialize()



# Parsed testcases at query #57
#--------------------------

# Failed to parse test__is_pclass_bases_returns_false.




# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields_raises_exception. Retrieved 1/10 statements.


def test_case_0():
    var_0 = -1



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_repr_contains_class_name_and_fields. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_check_and_set_attr_with_invalid_invariant. Retrieved 9/15 statements.


import builtins as module_0

def test_case_0():
    var_0 = None
    var_1 = 'MockClass'
    var_2 = ()
    var_3 = {}
    var_4 = 'test_field'
    var_5 = 'test_value'
    var_6 = module_0.object()
    var_7 = []
    var_8 = len(var_7)
    assert var_8 == 1



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/8 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 3/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = 2



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 2/8 statements.
# Failed to parse test_pclass_new_with_missing_mandatory_field.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_invalid_invariant. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 6/11 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Failed to parse test_pclass_new_with_callable_initial.


def test_case_0():
    var_0 = 1
    var_1 = 3

def test_case_0():
    var_0 = 'not an int'

def test_case_0():
    var_0 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'test'
    var_3 = 'extra'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 4



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_failed_invariant. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_string_type. Retrieved 4/12 statements.
# Partially parsed test_check_and_set_attr_with_multiple_types. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'attr'
    var_1 = 42
    var_2 = []

def test_case_0():
    var_0 = 'attr'
    var_1 = 'not_an_int'
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = 'attr'
    var_2 = 42
    var_3 = []

def test_case_0():
    var_0 = 'attr'
    var_1 = 'string'
    var_2 = []



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------




import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = module_1._PClassEvolver(var_0, var_5)
    var_7 = var_6.remove(var_1)

import builtins as module_0
import pyrsistent._pclass as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'a'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1._PClassEvolver(var_0, var_3)
    var_5 = 'b'
    var_6 = var_4.remove(var_5)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 7/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 0
    var_5 = None
    var_6 = 'test'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/10 statements.
# Partially parsed test_serialize_with_missing_optional_field. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'json'
    var_5 = module_0.serialize(var_4)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.serialize()



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_is_pclass_with_single_checkedtype_base.
# Failed to parse test_is_pclass_with_multiple_bases.
# Failed to parse test_is_pclass_with_non_checkedtype_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 3/7 statements.
# Failed to parse test_pclass_new_with_missing_mandatory_field.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Failed to parse test_pclass_new_with_initial_value.
# Failed to parse test_pclass_new_with_callable_initial.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 'not an int'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_inequality. Retrieved 6/12 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 2/9 statements.
# Partially parsed test_pclass_hash_with_different_types. Retrieved 3/10 statements.


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
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_eq_returns_true_for_equal_instances. Retrieved 4/7 statements.
# Partially parsed test_eq_returns_false_for_different_instances. Retrieved 5/8 statements.
# Partially parsed test_eq_returns_not_implemented_for_non_pclass. Retrieved 3/6 statements.


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
    var_2 = 'not a PClass'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

def test_case_0():
    var_0 = 'not an int'

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

def test_case_0():
    var_0 = 5
    var_1 = 'x'
    var_2 = {var_1}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclassmeta_new_with_checkedtype_base. Retrieved 5/13 statements.
# Partially parsed test_pclassmeta_new_without_checkedtype_base. Retrieved 5/15 statements.
# Partially parsed test_pclassmeta_new_with_inherited_invariants. Retrieved 2/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0._PField()
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0._PField()
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()

def test_case_0():
    var_0 = lambda self: True
    var_1 = lambda self: True

def test_case_0():
    var_0 = 'not callable'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()
    var_2 = module_0._PField()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_predicate_evaluates_to_false. Retrieved 3/5 statements.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = {}
    var_2 = {var_0: var_1}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_failed_invariant. Retrieved 3/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 'not an int'

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.
# Failed to parse test_pclass_constructor_with_initial_values.
# Partially parsed test_pclass_constructor_with_invalid_field_value. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = -1
    var_2 = 2



# Parsed testcases at query #16
#--------------------------




def test_case_0():
    var_0 = 'error1'
    var_1 = [var_0]
    var_2 = 'field1'
    var_3 = [var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_missing_field. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'json'
    var_4 = module_0.serialize(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = module_0.serialize()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 5/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 'custom'
    var_4 = module_0.serialize(var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = module_0.serialize()



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_equality_with_same_class_and_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #20
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = module_0.field()
    var_3 = lambda self: (False, 'error')
    var_4 = [var_3]
    var_5 = 1



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_eq_returns_true_for_identical_pclass_instances. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_check_and_set_attr_with_invalid_invariant. Retrieved 16/20 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = {}
    var_3 = 'MockField'
    var_4 = ()
    var_5 = 'invariant'
    var_6 = False
    var_7 = 'error'
    var_8 = (var_6, var_7)
    var_9 = lambda self, value: var_8
    var_10 = {var_5: var_9}
    var_11 = 'test_field'
    var_12 = 'test_value'
    var_13 = module_0.object()
    var_14 = []
    var_15 = hasattr(var_13, var_11)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_pclass_reduce_returns_restore_pickle_and_class_data_tuple. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pclass_hash_returns_consistent_value. Retrieved 4/9 statements.
# Partially parsed test_pclass_hash_different_instances_same_fields. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_different_fields. Retrieved 4/10 statements.
# Partially parsed test_pclass_hash_missing_optional_field. Retrieved 3/10 statements.
# Failed to parse test_pclass_hash_with_no_fields.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 0



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 1/8 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 6/12 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_with_valid_type. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_multiple_types. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

def test_case_0():
    var_0 = -1

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
    var_2 = 5
    var_3 = 3

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 'not_an_int'

def test_case_0():
    var_0 = 42
    var_1 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 'hello'
    var_4 = 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_factory_field. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 3/10 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 8/14 statements.
# Partially parsed test_pclass_new_with_type_check. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2

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
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 'hello'
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {var_1}
    var_7 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'not an int'
    var_2 = 2



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'



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

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_with_missing_field. Retrieved 6/11 statements.
# Partially parsed test_set_with_empty_kwargs. Retrieved 4/8 statements.


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
    var_4 = 30
    var_5 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invariant_violation. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 6/14 statements.
# Partially parsed test_pclass_new_with_global_invariant_violation. Retrieved 3/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/11 statements.
# Partially parsed test_pclass_new_with_type_check. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_multiple_types. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

def test_case_0():
    var_0 = -1

def test_case_0():
    var_0 = 'x'
    var_1 = 'z'
    var_2 = 'hello'
    var_3 = 'extra'
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = 5
    var_4 = 10

def test_case_0():
    var_0 = 'not_an_int'

def test_case_0():
    var_0 = 1
    var_1 = 'hello'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #35
#--------------------------

# Failed to parse test_invariant_errors_or_missing_fields_raises_exception.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/6 statements.
# Failed to parse test_pclass_new_with_missing_mandatory_field.
# Partially parsed test_pclass_new_with_invalid_field_type. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 6/8 statements.
# Failed to parse test_pclass_new_with_initial_value.
# Failed to parse test_pclass_new_with_callable_initial.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

def test_case_0():
    var_0 = 'not an int'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

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
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}

def test_case_0():
    var_0 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1

def test_case_0():
    var_0 = 'x'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pclass_hash_consistency. Retrieved 5/14 statements.
# Partially parsed test_pclass_hash_with_missing_fields. Retrieved 3/10 statements.
# Partially parsed test_pclass_hash_with_different_field_order. Retrieved 4/10 statements.


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
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_multiple_updates. Retrieved 8/12 statements.
# Partially parsed test_set_with_missing_field. Retrieved 7/12 statements.
# Partially parsed test_set_with_mandatory_field_missing. Retrieved 4/9 statements.
# Partially parsed test_set_with_initial_value_field. Retrieved 3/8 statements.
# Partially parsed test_set_with_callable_initial_value_field. Retrieved 3/8 statements.


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
    var_1 = 1
    var_2 = 2
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2
    var_2 = 10



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #43
#--------------------------

# Partially parsed test__is_pclass_bases_returns_false. Retrieved 4/6 statements.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = {}
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_check_and_set_attr_invariant_false. Retrieved 13/16 statements.


import builtins as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'invariant'
    var_3 = False
    var_4 = 'error'
    var_5 = (var_3, var_4)
    var_6 = lambda x: var_5
    var_7 = {var_2: var_6}
    var_8 = 'test_field'
    var_9 = 'test_value'
    var_10 = module_0.object()
    var_11 = []
    var_12 = hasattr(var_10, var_8)



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 1



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_invariant. Retrieved 4/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 'attr_name'
    var_2 = 'value'
    var_3 = []



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_pclass_meta_new_with_single_checked_type_base. Retrieved 7/10 statements.
# Partially parsed test_pclass_meta_new_with_multiple_bases. Retrieved 5/12 statements.
# Partially parsed test_pclass_meta_new_with_invariant. Retrieved 8/4 statements.
# Partially parsed test_pclass_meta_new_with_inherited_fields_and_invariants. Retrieved 6/11 statements.
# Partially parsed test_pclass_meta_new_with_non_pclass_bases. Retrieved 4/8 statements.
# Partially parsed test_pclass_meta_new_with_pfield_in_dct. Retrieved 6/8 statements.
# Partially parsed test_pclass_meta_new_with_invalid_invariant_type. Retrieved 6/9 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'TestClass'
    var_6 = '_pclass_invariants'

def test_case_0():
    var_0 = 'c'
    var_1 = 3
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'
    var_4 = '_pclass_invariants'

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'x'
    var_5 = 10
    var_6 = 'TestClass'
    var_7 = 0

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'x'
    var_5 = 10
    var_6 = 'TestClass'
    var_7 = 0

def test_case_0():
    var_0 = lambda self: (True, 'Parent invariant')
    var_1 = 1
    var_2 = 'child_field'
    var_3 = 2
    var_4 = {var_2: var_3}
    var_5 = 'ChildClass'

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'regular'
    var_1 = '_pfield'
    var_2 = 1
    var_3 = module_0._PField()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'TestClass'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'field'
    var_2 = 'not callable'
    var_3 = 1
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'TestClass'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 2/8 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 1/11 statements.
# Partially parsed test_pclass_constructor_with_valid_invariant. Retrieved 1/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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

def test_case_0():
    var_0 = -1

def test_case_0():
    var_0 = 1



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_pickle_support_returns_correct_tuple. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_pclassmeta_new_with_checkedtype_base. Retrieved 3/8 statements.
# Partially parsed test_pclassmeta_new_without_checkedtype_base. Retrieved 3/8 statements.
# Failed to parse test_pclassmeta_new_with_invariants.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'

def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = '_pclass_invariants'
    var_2 = '__slots__'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()

def test_case_0():
    var_0 = 'not callable'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_pclassmeta_new_with_checkedtype_base. Retrieved 15/19 statements.
# Partially parsed test_pclassmeta_new_without_checkedtype_base. Retrieved 12/18 statements.
# Partially parsed test_pclassmeta_new_with_inherited_invariants. Retrieved 12/19 statements.


import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = '_pclass_fields'
    var_4 = '_pclass_invariants'
    var_5 = '__slots__'
    var_6 = True
    var_7 = lambda self: var_6
    var_8 = module_1.wrap_invariant(var_7)
    var_9 = (var_8,)
    var_10 = 'field1'
    var_11 = 'field2'
    var_12 = module_0.field()
    var_13 = module_0.field()
    var_14 = {var_10: var_12, var_11: var_13}

import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0.field()
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'
    var_5 = True
    var_6 = lambda self: var_5
    var_7 = module_1.wrap_invariant(var_6)
    var_8 = (var_7,)
    var_9 = 'field1'
    var_10 = module_0.field()
    var_11 = {var_9: var_10}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda self: True
    var_1 = module_0.field()
    var_2 = lambda self: False
    var_3 = module_0.field()
    var_4 = '_pclass_fields'
    var_5 = '_pclass_invariants'
    var_6 = '__slots__'
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = module_0.field()
    var_10 = module_0.field()
    var_11 = {var_7: var_9, var_8: var_10}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'not callable'
    var_1 = module_0.field()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 4/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

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
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

def test_case_0():
    var_0 = -1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2

def test_case_0():
    var_0 = 'x'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = True



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
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
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

def test_case_0():
    var_0 = -1

def test_case_0():
    var_0 = 'x'
    var_1 = 'hello'
    var_2 = {var_0: var_1}
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field. Retrieved 1/5 statements.
# Failed to parse test_pclass_constructor_with_missing_mandatory_field.
# Failed to parse test_pclass_constructor_with_initial_value.
# Failed to parse test_pclass_constructor_with_callable_initial.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 2/9 statements.
# Partially parsed test_pclass_constructor_with_invariant_violation. Retrieved 2/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

def test_case_0():
    var_0 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

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
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = {var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_failing_invariant. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_string_type. Retrieved 4/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_string_type. Retrieved 4/13 statements.
# Partially parsed test_check_and_set_attr_with_multiple_types. Retrieved 4/15 statements.
# Partially parsed test_check_and_set_attr_with_invalid_multiple_types. Retrieved 3/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 'not_an_int'

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = []
    var_2 = 'attr'
    var_3 = 42

def test_case_0():
    var_0 = 'builtins.int'
    var_1 = []
    var_2 = 'attr'
    var_3 = 'not_an_int'

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42
    var_3 = 'test'

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 3.14



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/5 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 3/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

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
    var_1 = 5
    var_2 = 15



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 4/13 statements.


def test_case_0():
    var_0 = None
    var_1 = []
    var_2 = 'attr'
    var_3 = 'value'



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 2/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 2/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3

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
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2



