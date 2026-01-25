####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pclass_meta_new_with_single_checkedtype_base. Retrieved 3/8 statements.
# Partially parsed test_pclass_meta_new_with_multiple_bases. Retrieved 3/12 statements.
# Failed to parse test_pclass_meta_new_with_field_inheritance.
# Partially parsed test_pclass_meta_new_with_invariant_inheritance. Retrieved 2/12 statements.


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
    var_0 = 0
    var_1 = None

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test___reduce___returns_correct_tuple. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_pickle_support_returns_correct_tuple. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/3 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/3 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 2/4 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 3/9 statements.


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
    var_0 = 0
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = 5

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type for field TestClass.x'

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
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 1
    var_4 = 2
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v.upper()
    var_1 = module_0.field(factory=var_0)
    var_2 = 'hello'
    var_3 = 'x'
    var_4 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'Global invariant failed'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_mandatory_field_missing. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/10 statements.
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
    var_0 = lambda x: x > 0
    var_1 = module_0.field(invariant=var_0)
    var_2 = module_0.field()
    var_3 = -1
    var_4 = 2
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Field invariant failed'

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
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/7 statements.
# Partially parsed test_serialize_with_missing_fields. Retrieved 3/6 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/8 statements.


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
    var_0 = lambda v, fmt: f'{v}_{fmt}' if fmt else str(v)
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 'json'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test__is_pclass_with_single_checkedtype_base.
# Failed to parse test__is_pclass_with_multiple_bases.
# Failed to parse test__is_pclass_with_non_checkedtype_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_pclass_repr. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_serialize_returns_dict_with_correct_keys. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #10
#--------------------------

# Failed to parse test__is_pclass_returns_true_for_pclass_bases.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 4/12 statements.


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
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_pclassmeta_new_with_checkedtype_base. Retrieved 4/13 statements.
# Partially parsed test_pclassmeta_new_without_checkedtype_base. Retrieved 4/14 statements.
# Partially parsed test_pclassmeta_new_with_inherited_fields. Retrieved 4/14 statements.
# Partially parsed test_pclassmeta_new_with_inherited_invariants. Retrieved 5/15 statements.


def test_case_0():
    var_0 = lambda self: True
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = '__slots__'

def test_case_0():
    var_0 = lambda self: True
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = '__slots__'

def test_case_0():
    var_0 = lambda self: True
    var_1 = '_pclass_fields'
    var_2 = '_pclass_invariants'
    var_3 = '__slots__'

def test_case_0():
    var_0 = lambda self: True
    var_1 = lambda self: False
    var_2 = '_pclass_fields'
    var_3 = '_pclass_invariants'
    var_4 = '__slots__'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #13
#--------------------------




def test_case_0():
    var_0 = 'error1'
    var_1 = [var_0]
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = bool(var_1 or var_3)
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 1/8 statements.
# Partially parsed test_pclass_constructor_with_global_invariant. Retrieved 4/11 statements.
# Partially parsed test_pclass_constructor_with_valid_invariant. Retrieved 1/7 statements.


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

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 4
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Field invariant failed'

def test_case_0():
    var_0 = 1



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()



# Parsed testcases at query #16
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



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/8 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.


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
    var_2 = 1
    var_3 = 2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum'

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



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 1/9 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_factory_and_ignore_extra. Retrieved 4/10 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_type_check_failure. Retrieved 1/7 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 5/11 statements.


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
    var_5 = 'TestClass.y'
    var_6 = (var_5,)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = "'z' are not among the specified fields for TestClass"

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
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 1

def test_case_0():
    var_0 = -1
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'positive'
    var_3 = (var_2,)

def test_case_0():
    var_0 = 5

def test_case_0():
    var_0 = 'x'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum_positive'
    var_6 = (var_5,)

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type for field TestClass.x, was str'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 1
    var_3 = 'x'
    var_4 = {var_3}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_set_new_key_marks_dirty. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_with_different_value_marks_dirty. Retrieved 4/7 statements.
# Partially parsed test_set_existing_key_with_same_value_does_not_mark_dirty. Retrieved 3/6 statements.
# Partially parsed test_set_returns_self. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'new_key'
    var_3 = 'value'
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



# Parsed testcases at query #20
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



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
# Partially parsed test_set_with_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_set_with_missing_field. Retrieved 7/12 statements.


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



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_new_with_invariant_error. Retrieved 2/7 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/12 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_pclass_new_with_type_check. Retrieved 1/7 statements.


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
    var_2 = module_0.field()
    var_3 = 5
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v, ignore_extra=False: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = 'x'
    var_3 = 'z'
    var_4 = 5
    var_5 = 3
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True

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
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = module_0.field()
    var_3 = 5
    var_4 = 2
    var_5 = 'x'
    var_6 = {var_5}

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.


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

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type for field'

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
    var_2 = 21
    var_3 = 'x'
    var_4 = {var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Global invariant failed'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_pclass_eq_with_same_instance. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_with_different_instance. Retrieved 6/10 statements.
# Partially parsed test_pclass_eq_with_different_class. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_with_non_pclass_instance. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_with_missing_fields. Retrieved 4/8 statements.


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
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_repr_returns_correct_format. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields_raises_exception. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #30
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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_set_new_key_marks_dirty. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_with_different_value_marks_dirty. Retrieved 4/7 statements.
# Partially parsed test_set_existing_key_with_same_value_does_not_mark_dirty. Retrieved 3/6 statements.
# Partially parsed test_set_returns_self. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'new_key'
    var_3 = 'value'
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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_hash_equality. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #33
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #34
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #35
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #36
#--------------------------

# Partially parsed test_set_predicate_false. Retrieved 3/8 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = []



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/11 statements.


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

def test_case_0():
    var_0 = 'not an int'
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Invalid type'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v: v * 2
    var_1 = module_0.field(factory=var_0)
    var_2 = 5

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
    var_5 = 'Global invariant failed'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_set_predicate_false. Retrieved 3/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = []



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 10/13 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 6/8 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = {var_2}
    var_4 = 1
    var_5 = 2



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pclass_pickling_returns_correct_tuple. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #43
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



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_check_and_set_attr_with_valid_type_and_invariant. Retrieved 3/12 statements.
# Partially parsed test_check_and_set_attr_with_invalid_type. Retrieved 3/13 statements.
# Partially parsed test_check_and_set_attr_with_failed_invariant. Retrieved 3/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42
    var_3 = bool(var_0 == [])
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 'not_an_int'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = []
    var_1 = 'attr'
    var_2 = 42
    var_3 = bool(var_0 == ['INVALID'])
    assert var_3 is True



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 1
    var_5 = 2
    var_6 = 3



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/5 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 1/8 statements.


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
    var_0 = 5
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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3

def test_case_0():
    var_0 = 15
    var_1 = bool(False)
    assert var_1 is True
    var_2 = 'Field invariant failed'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_pclassmeta_new_with_checkedtype_base. Retrieved 3/8 statements.
# Partially parsed test_pclassmeta_new_without_checkedtype_base. Retrieved 3/10 statements.
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



# Parsed testcases at query #3
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



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_set_new_key_value_pair. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_with_same_value. Retrieved 3/6 statements.
# Partially parsed test_set_existing_key_with_different_value. Retrieved 4/7 statements.
# Partially parsed test_set_returns_self. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'
    var_4 = 'key'

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'key'

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'old_value'
    var_3 = {var_1: var_2}
    var_4 = 'new_value'
    var_5 = 'key'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_repr_with_no_fields.
# Partially parsed test_repr_with_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_with_multiple_fields. Retrieved 4/7 statements.
# Partially parsed test_repr_with_missing_optional_field. Retrieved 1/4 statements.
# Partially parsed test_repr_with_complex_field_values. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = module_0.field(mandatory=var_0)

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



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_is_pclass_with_single_checkedtype_base.
# Failed to parse test_is_pclass_with_multiple_bases.
# Failed to parse test_is_pclass_with_non_checkedtype_base.


import pyrsistent._pclass as module_0

def test_case_0():
    var_0 = ()
    var_1 = module_0._is_pclass(var_0)
    assert var_1 is False



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_factory. Retrieved 2/5 statements.
# Partially parsed test_pclass_new_with_invariant. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/11 statements.


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
    var_0 = 5
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
    var_2 = 3

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
    var_0 = lambda v, ignore_extra=False: v
    var_1 = module_0.field(factory=var_0)
    var_2 = 'x'
    var_3 = 'z'
    var_4 = 1
    var_5 = 3
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 3
    var_3 = 4
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum_10'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 4/8 statements.
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
    var_0 = lambda v: v * 2
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 5
    var_4 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda v, fmt: str(v) if fmt == 'str' else v
    var_1 = module_0.field(serializer=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'test'
    var_5 = 'str'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_mandatory_field_missing. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
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
    var_3 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_pclass_hash_equality. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_serialize_includes_all_fields. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #12
#--------------------------

# Failed to parse test__is_pclass_returns_false_for_non_pclass_bases.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serialize_returns_dict_with_field_names_and_serialized_values. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda v: v * 2
    var_2 = module_0.field(serializer=var_1)
    var_3 = 1
    var_4 = 2
    var_5 = 'x'
    var_6 = 'y'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_pclassmeta_new_sets_fields. Retrieved 1/5 statements.
# Partially parsed test_pclassmeta_new_stores_invariants. Retrieved 1/8 statements.
# Failed to parse test_pclassmeta_new_sets_slots.


def test_case_0():
    var_0 = '_pclass_fields'
    var_1 = 'field1'
    var_2 = 'field2'

def test_case_0():
    var_0 = '_pclass_invariants'

def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_pclass_reduce. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_hash_returns_consistent_value. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_set_new_key_marks_dirty_and_adds_to_factory_fields. Retrieved 3/6 statements.
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

def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = 'key'
    var_3 = 'value'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 2/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = lambda v: (False, 'error')
    var_3 = module_0.field(invariant=var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.


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



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 8/14 statements.


def test_case_0():
    var_0 = 'error'
    var_1 = [var_0]
    var_2 = []
    var_3 = bool(var_1 or var_2)
    assert var_3 is True
    var_4 = []
    var_5 = 'TestClass.x'
    var_6 = [var_5]
    var_7 = bool(var_4 or var_6)
    assert var_7 is True
    var_8 = [var_0]
    var_9 = [var_5]
    var_10 = bool(var_8 or var_9)
    assert var_10 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_pickle_support_returns_correct_tuple. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_pclass_fields_iteration. Retrieved 2/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '_pclass_fields'
    var_2 = 'items'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_remove_existing_item. Retrieved 5/8 statements.
# Partially parsed test_remove_nonexistent_item_raises_error. Retrieved 4/8 statements.


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



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_invariant_errors_or_missing_fields. Retrieved 10/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = []
    var_3 = 'error1'
    var_4 = [var_3]
    var_5 = []
    var_6 = bool(var_4 or var_5)
    assert var_6 is True
    var_7 = []
    var_8 = 'field1'
    var_9 = [var_8]
    var_10 = bool(var_7 or var_9)
    assert var_10 is True
    var_11 = [var_3]
    var_12 = [var_8]
    var_13 = bool(var_11 or var_12)
    assert var_13 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_set_with_keyword_arguments. Retrieved 5/9 statements.
# Partially parsed test_set_with_positional_arguments. Retrieved 6/10 statements.
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



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.


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
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

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



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_pclassmeta_new_with_single_checkedtype_base. Retrieved 3/8 statements.
# Partially parsed test_pclassmeta_new_with_multiple_bases. Retrieved 3/12 statements.
# Partially parsed test_pclassmeta_new_inherits_fields_and_invariants. Retrieved 3/7 statements.


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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda self: True
    var_2 = module_0.field()
    var_3 = 'x'
    var_4 = 'y'

def test_case_0():
    var_0 = 'not callable'
    var_1 = bool(False)
    assert var_1 is True



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_pclass_eq_same_instance. Retrieved 4/7 statements.
# Partially parsed test_pclass_eq_different_instances_same_values. Retrieved 4/8 statements.
# Partially parsed test_pclass_eq_different_values. Retrieved 6/10 statements.
# Partially parsed test_pclass_eq_different_classes. Retrieved 3/8 statements.
# Partially parsed test_pclass_eq_non_pclass_instance. Retrieved 2/5 statements.
# Partially parsed test_pclass_eq_missing_fields. Retrieved 3/7 statements.


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
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_eq_with_same_class_and_values. Retrieved 4/8 statements.
# Partially parsed test_eq_with_different_class. Retrieved 3/8 statements.
# Partially parsed test_eq_with_different_values. Retrieved 5/9 statements.
# Partially parsed test_eq_with_missing_fields. Retrieved 4/8 statements.
# Partially parsed test_eq_with_non_pclass_object. Retrieved 2/5 statements.


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
    var_1 = False
    var_2 = module_0.field(mandatory=var_1)
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_pclass_pickling_returns_correct_tuple. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test_pclass_new_with_invalid_type. Retrieved 1/6 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 2/6 statements.
# Partially parsed test_pclass_new_with_global_invariant_failure. Retrieved 4/10 statements.


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
    var_4 = 'y'

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
    var_4 = 'must be positive'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'Global invariant failed'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_serialize_includes_all_non_missing_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'a'
    var_7 = 'b'
    var_8 = 'c'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_factory_fields. Retrieved 8/11 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_invariant_check. Retrieved 2/10 statements.
# Partially parsed test_pclass_constructor_with_pclass_instance. Retrieved 4/8 statements.


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
    var_3 = 'invariant'
    var_4 = bool('invariant' in str(e).lower())
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_eq_returns_true_for_identical_pclass_instances. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_pclass_new_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_new_with_missing_mandatory_field. Retrieved 3/8 statements.
# Partially parsed test_pclass_new_with_invalid_field_type. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_invariant_failure. Retrieved 3/11 statements.
# Partially parsed test_pclass_new_with_extra_fields. Retrieved 5/9 statements.
# Partially parsed test_pclass_new_with_ignore_extra. Retrieved 10/14 statements.
# Partially parsed test_pclass_new_with_factory_fields. Retrieved 3/9 statements.
# Partially parsed test_pclass_new_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_new_with_callable_initial. Retrieved 2/8 statements.
# Partially parsed test_pclass_new_with_global_invariant. Retrieved 4/12 statements.


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
    var_1 = 'not_an_int'
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = -1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'must_be_positive'

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
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = -1
    var_3 = -2
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'sum_must_be_positive'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_remove_item_exists. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'key'
    var_5 = 'key'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_set_preserves_existing_fields. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 10



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_pclass_constructor_with_valid_fields. Retrieved 4/7 statements.
# Partially parsed test_pclass_constructor_with_missing_mandatory_field. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_extra_fields. Retrieved 3/7 statements.
# Partially parsed test_pclass_constructor_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_pclass_constructor_with_initial_value. Retrieved 3/6 statements.
# Partially parsed test_pclass_constructor_with_callable_initial. Retrieved 3/6 statements.


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
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True

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



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    var_0 = '__weakref__'



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    var_0 = 'error1'
    var_1 = [var_0]
    var_2 = 'field1'
    var_3 = [var_2]
    var_4 = bool(var_1 or var_3)
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_repr_format. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



