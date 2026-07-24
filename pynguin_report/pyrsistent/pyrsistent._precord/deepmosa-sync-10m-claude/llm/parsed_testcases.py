####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 5/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 7/20 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/19 statements.
# Partially parsed test_persistent_returns_same_instance_when_not_dirty. Retrieved 5/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockCls'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockCls.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = {}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.pmap(var_3)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_slots_empty. Retrieved 6/12 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 7/19 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 8/17 statements.
# Partially parsed test_precord_meta_new_removes_field_from_dct. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default'
    var_3 = False
    var_4 = True
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default'
    var_3 = False
    var_4 = True
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_mandatory_fields'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default'
    var_3 = False
    var_4 = True
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_initial_values'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'default'
    var_2 = False
    var_3 = ()
    var_4 = 'TestRecord'
    var_5 = '__slots__'

def test_case_0():
    var_0 = 'field1'
    var_1 = '__invariant__'
    var_2 = 'default'
    var_3 = False
    var_4 = ()
    var_5 = 'TestRecord'
    var_6 = '_precord_invariants'

def test_case_0():
    var_0 = 'base_field'
    var_1 = 'base'
    var_2 = False
    var_3 = 'BaseRecord'
    var_4 = ()
    var_5 = 'child_field'
    var_6 = True
    var_7 = 'ChildRecord'
    var_8 = 'base_field'
    var_9 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'default'
    var_2 = False
    var_3 = ()
    var_4 = 'TestRecord'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 5/11 statements.
# Partially parsed test_precord_evolver_set_with_invalid_field_raises_attribute_error. Retrieved 6/13 statements.
# Partially parsed test_precord_evolver_set_with_type_check_failure. Retrieved 6/13 statements.
# Partially parsed test_precord_evolver_setitem_delegates_to_set. Retrieved 5/11 statements.
# Partially parsed test_precord_evolver_set_with_factory_field. Retrieved 5/11 statements.
# Partially parsed test_precord_evolver_set_with_restricted_factory_fields. Retrieved 7/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'new_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'invalid_field'
    var_5 = 'value'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'invalid_field'
    var_8 = 'TestRecord'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = 25
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'age'
    var_5 = 'not_an_int'
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'test'
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = 'updated'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = 5
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = '10'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'test'
    var_3 = 25
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'updated'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 1/8 statements.
# Failed to parse test_precord_new_empty.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/8 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/10 statements.
# Partially parsed test_precord_new_overrides_initial_values. Retrieved 2/6 statements.


def test_case_0():
    var_0 = 0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 100

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = var_1[0]
    assert var_2 == 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_precord_evolver_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 2/11 statements.
# Partially parsed test_precord_evolver_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 2/11 statements.
# Partially parsed test_precord_evolver_persistent_raises_invariant_exception_when_both_present. Retrieved 4/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error_code_1'
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'TestRecord.name'
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error_code_1'
    var_2 = 'error_code_2'
    var_3 = 'TestRecord.name'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 4/13 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_with_mixed_serializers. Retrieved 4/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'john'
    var_2 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'
    var_2 = 'data'
    var_3 = 'uppercase'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'hello'
    var_3 = 'unchanged'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42
    var_4 = 'name'
    var_5 = 'value'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_precord_mandatory_fields_exist. Retrieved 6/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = []
    var_5 = 'John'
    var_6 = 30



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_precord_meta_new_returns_class. Retrieved 4/11 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_field'
    var_2 = 'TestRecord'
    var_3 = ()



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : []
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 2/15 statements.
# Partially parsed test_set_with_nonexistent_field_raises_attribute_error. Retrieved 3/12 statements.
# Partially parsed test_set_with_factory_fields_filter. Retrieved 2/17 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 2/14 statements.
# Partially parsed test_set_with_factory_exception. Retrieved 2/17 statements.
# Partially parsed test_setitem_delegates_to_set. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'test_value'

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = 'nonexistent_field'
    var_3 = 'value'
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'nonexistent_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = []
    var_3 = 'field1'
    var_4 = 'value1'

def test_case_0():
    var_0 = 'test_field'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'test_value'
    var_4 = 'error_code'

def test_case_0():
    var_0 = 'test_field'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'test_value'
    var_4 = 'error1'
    var_5 = 'missing1'

def test_case_0():
    var_0 = 'test_field'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 'test_value'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 3/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 3/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_1.pmap()
    var_2 = 'error1'
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_1.pmap()
    var_2 = 'TestRecord.x'
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_1.pmap()
    var_2 = 'error1'
    var_3 = 'error2'
    var_4 = 'TestRecord.x'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_precord_initial_values_predicate. Retrieved 8/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = lambda : 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 10
    var_10 = lambda : var_9
    var_11 = 20



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_slots. Retrieved 5/14 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'test_field'
    var_3 = 'TestRecord'
    var_4 = ()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_slots. Retrieved 8/14 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'test_attr'
    var_3 = '__module__'
    var_4 = '__qualname__'
    var_5 = '__main__'
    var_6 = 'TestPRecord'
    var_7 = ()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_persistent_returns_result_when_clean_and_correct_type. Retrieved 8/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 10/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 8/15 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 7/17 statements.
# Partially parsed test_persistent_with_passing_global_invariants. Retrieved 7/19 statements.
# Partially parsed test_persistent_with_dirty_state_creates_new_instance. Retrieved 8/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 'error1'
    var_9 = 'error2'
    var_10 = bool(False)
    assert var_10 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockClass'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = bool(False)
    assert var_10 is True
    var_11 = 'MockClass.field1'
    var_12 = 'MockClass.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_persistent_predicate_is_dirty_true. Retrieved 6/11 statements.
# Partially parsed test_persistent_predicate_not_isinstance_true. Retrieved 7/13 statements.
# Partially parsed test_persistent_predicate_both_conditions_true. Retrieved 8/16 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'modified'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = {var_1: var_2}
    var_6 = module_1.pmap(var_5)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'initial'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'changed'
    var_6 = {var_1: var_5}
    var_7 = module_1.pmap(var_6)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 4/7 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 'Jane'
    var_5 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Test'
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Test'
    var_2 = True
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'Test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_special_characters. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord'
    var_5 = 'name='
    var_6 = "'Alice'"
    var_7 = 'age='
    var_8 = '30'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 'SingleFieldRecord'
    var_3 = 'value=42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = "hello'world"
    var_2 = 'SpecialRecord'
    var_3 = 'text='
    var_4 = "hello'world"



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 2/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 2/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present. Retrieved 4/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error1'
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'TestRecord.x'
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error1'
    var_2 = 'error2'
    var_3 = 'TestRecord.x'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_persistent_predicate_is_dirty_true. Retrieved 3/13 statements.
# Partially parsed test_persistent_predicate_not_isinstance_true. Retrieved 5/13 statements.
# Partially parsed test_persistent_predicate_is_dirty_false_isinstance_false. Retrieved 2/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'x'
    var_3 = 1

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_precord_evolver_persistent_predicate_is_dirty_true. Retrieved 10/19 statements.
# Partially parsed test_precord_evolver_persistent_predicate_not_isinstance_true. Retrieved 9/17 statements.
# Partially parsed test_precord_evolver_persistent_predicate_both_conditions_true. Retrieved 11/22 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = 10

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = 20
    var_9 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_with_callable_initial.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/11 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 8/11 statements.
# Partially parsed test_precord_constructor_with_none_values. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._HAMT_MUTATION_TRACKING_ENABLED
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_precord_fields. Retrieved 7/15 statements.


def test_case_0():
    var_0 = True
    var_1 = 42
    var_2 = False
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = ()
    var_6 = 'TestClass'
    var_7 = '_precord_fields'
    var_8 = '_precord_mandatory_fields'
    var_9 = '_precord_initial_values'
    var_10 = '_precord_invariants'
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'field1'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 4/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 5/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_errors. Retrieved 7/20 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 8/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = {}
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2, var_3}
    var_5 = []
    var_6 = module_0.pmap()
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockClass.field1'
    var_9 = 'MockClass.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = module_0.pmap()
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error1'
    var_9 = 'error2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = 'MockClass'
    var_2 = {}
    var_3 = set()
    var_4 = module_0.pmap()
    var_5 = len(var_0)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_0[var_6]



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_precord_repr. Retrieved 6/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord'
    var_5 = 'name='
    var_6 = "'Alice'"
    var_7 = 'age='
    var_8 = '30'
    var_9 = 'TestRecord('
    var_10 = ')'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_present. Retrieved 4/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = []



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_new_partial_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_callable_initial_value. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_multiple_kwargs. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = var_7._size
    var_9 = var_7._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 15
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 15
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 3
    var_5 = 'y'

def test_case_0():
    var_0 = 0
    var_1 = [var_0]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_precord_evolver_set_with_field_found. Retrieved 7/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_key'
    var_2 = set()
    var_3 = ()
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'test_key'
    var_7 = 42



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 8/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 8/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields. Retrieved 10/19 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'error1'
    var_6 = False
    var_7 = True
    assert var_7 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'TestRecord.y'
    var_6 = False
    var_7 = True
    assert var_7 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = 'TestRecord.y'
    var_8 = False
    var_9 = True
    assert var_9 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields_combined. Retrieved 5/14 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 3/14 statements.
# Partially parsed test_persistent_succeeds_with_no_errors. Retrieved 3/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockClass'
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'MockClass.field1'

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockClass'
    var_6 = []
    var_7 = 'field0'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'field0'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = []



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_precord_evolver_set_with_field_found. Retrieved 2/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 'name'
    var_2 = 'test_value'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_slots. Retrieved 8/17 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_field'
    var_2 = 'TestClass'
    var_3 = ()
    var_4 = '_precord_fields'
    var_5 = '_precord_invariants'
    var_6 = '_precord_mandatory_fields'
    var_7 = '_precord_initial_values'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_precord_repr. Retrieved 5/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord('
    var_5 = "name='Alice'"
    var_6 = 'age=30'
    var_7 = ')'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_precord_mandatory_fields_exists. Retrieved 4/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = []
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_empty_kwargs. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_overrides_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = var_7._size
    var_9 = var_7._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.field(initial=var_0)
    var_2 = 200
    var_3 = module_0.field(initial=var_2)

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = var_1[0]
    assert var_2 == 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = False

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.field(initial=var_0)
    var_2 = 200
    var_3 = module_0.field(initial=var_2)
    var_4 = 50

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_initial_values_callable. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_initial_values_override. Retrieved 3/6 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_new_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = var_7._size
    var_9 = var_7._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = lambda : 42
    var_2 = {var_0: var_1}
    var_3 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = 200

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = True

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

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 42
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_persistent_predicate_line_6_true_when_is_dirty. Retrieved 2/14 statements.
# Partially parsed test_persistent_predicate_line_6_true_when_not_isinstance. Retrieved 2/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = True
    var_3 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = False
    var_3 = []



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/15 statements.
# Partially parsed test_persistent_raises_on_invariant_error_codes. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_on_missing_mandatory_fields. Retrieved 4/14 statements.
# Partially parsed test_persistent_raises_on_missing_fields. Retrieved 5/15 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = 'error1'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockClass'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = 'missing_field'
    var_6 = bool(False)
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'MockClass'
    var_4 = []
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(var_0)
    assert var_6 is True
    var_7 = 'global_error'



# Parsed testcases at query #42
#--------------------------

# Failed to parse test_repr_empty_record.
# Partially parsed test_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 4/8 statements.
# Partially parsed test_repr_with_integer_values. Retrieved 2/6 statements.
# Partially parsed test_repr_with_string_values. Retrieved 2/6 statements.
# Partially parsed test_repr_with_none_value. Retrieved 2/6 statements.
# Partially parsed test_repr_with_boolean_values. Retrieved 2/6 statements.
# Partially parsed test_repr_with_list_value. Retrieved 5/9 statements.
# Partially parsed test_repr_with_nested_record. Retrieved 3/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'MultiFieldRecord('
    var_5 = "name='John'"
    var_6 = 'age=30'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello world'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'OuterRecord('
    var_4 = 'InnerRecord(inner_value=10)'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/24 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 2/23 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 3/25 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 2/27 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = ()
    var_4 = (var_3,)
    var_5 = 0
    var_6 = []

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = []
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'MockClass.field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = 'error_code_1'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'error_code_1'

def test_case_0():
    var_0 = []
    var_1 = 'field1'
    var_2 = set()
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True
    var_5 = bool(var_0)
    assert var_5 is True
    var_6 = 'global_error'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42
    var_4 = 'name'
    var_5 = 'value'



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_precord_meta_new_sets_fields. Retrieved 7/17 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 6/16 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 5/15 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 3/12 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 8/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 'default'
    var_5 = ()
    var_6 = 'TestClass'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = ()
    var_5 = 'TestClass'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default_value'
    var_3 = ()
    var_4 = 'TestClass'

def test_case_0():
    var_0 = 'field1'
    var_1 = ()
    var_2 = 'TestClass'

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = ()
    var_6 = 'TestClass'
    var_7 = '_precord_invariants'
    var_8 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = ()
    var_6 = 'TestClass'
    var_7 = '_precord_invariants'
    var_8 = '_precord_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 'BaseClass'
    var_3 = ()
    var_4 = 'field2'
    var_5 = False
    var_6 = 'ChildClass'
    var_7 = 'field1'
    var_8 = 'field2'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_exist. Retrieved 3/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_exist. Retrieved 2/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_exist. Retrieved 3/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = []
    var_3 = 'error_code_1'
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'TestRecord.name'
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'error_code_1'
    var_3 = 'TestRecord.name'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42
    var_4 = 'name'
    var_5 = 'value'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = var_7._size
    var_9 = var_7._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda : 42
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.field(initial=var_0)
    var_2 = 200
    var_3 = module_0.field(initial=var_2)
    var_4 = 10
    var_5 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = set()
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 999
    var_3 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_partial_kwargs_and_initial_values. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'
    var_2 = 'value'
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'
    var_2 = 'value'
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_precord_new_predicate_false_when_missing_precord_size. Retrieved 3/7 statements.
# Partially parsed test_precord_new_predicate_false_when_missing_precord_buckets. Retrieved 3/7 statements.
# Partially parsed test_precord_new_predicate_false_when_both_missing. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 3



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_kwargs_overrides_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : []
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : {}
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_precord_fields. Retrieved 12/25 statements.


def test_case_0():
    var_0 = True
    var_1 = 42
    var_2 = False
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = '_precord_mandatory_fields'
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = '_precord_initial_values'
    var_14 = 'field1'
    var_15 = '__slots__'
    var_16 = '_precord_invariants'



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_precord_new_predicate_false_without_precord_size. Retrieved 4/7 statements.
# Partially parsed test_precord_new_predicate_false_without_precord_buckets. Retrieved 4/7 statements.
# Partially parsed test_precord_new_predicate_false_with_regular_kwargs. Retrieved 4/7 statements.


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
    var_2 = 5
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 6/16 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 0



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/19 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 2
    var_3 = b'x'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = b'y'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]
    var_9 = [var_8]



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_partial_kwargs. Retrieved 5/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 'Alice'
    var_5 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Bob'
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Charlie'
    var_2 = 'ignored'
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'David'
    var_4 = 40



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_partial_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_kwargs_override_initial. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 7/12 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100
    var_5 = 200

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 'z'
    var_3 = 1
    var_4 = 3
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 42
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 7/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 9/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/20 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 7/18 statements.
# Partially parsed test_persistent_with_both_error_codes_and_missing_fields. Retrieved 7/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = 'error1'
    var_8 = 'error2'
    var_9 = bool(False)
    assert var_9 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = {var_2, var_3}
    var_5 = []
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'MockCls.field1'
    var_10 = 'MockCls.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = {}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = 'required_field'
    var_3 = {var_2}
    var_4 = []
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = 'inv_error'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'inv_error'
    var_10 = 'MockCls.required_field'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_persistent_predicate_line_6. Retrieved 4/19 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'test'
    var_4 = []



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values_override. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_new_empty_record.
# Partially parsed test_precord_new_with_multiple_fields. Retrieved 8/11 statements.


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
    var_2 = 'Alice'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = []
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 999
    var_3 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.field()
    var_6 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.field()
    var_6 = module_0.field()
    var_7 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = lambda : 42
    var_2 = {var_0: var_1}
    var_3 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_persistent_evaluates_predicate_at_line_1. Retrieved 8/15 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'value'
    var_4 = 'test'
    var_5 = 42
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 5/28 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = False
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 7/21 statements.
# Partially parsed test_persistent_raises_on_invariant_error_codes. Retrieved 6/17 statements.
# Partially parsed test_persistent_raises_on_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 4/19 statements.
# Partially parsed test_persistent_not_dirty_and_same_type. Retrieved 6/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = 'error1'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockClass'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockClass.field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = {}
    var_3 = module_0.pmap(var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = {}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = '_precord_size'
    var_5 = bool('_precord_size' not in {'name': 'John', 'age': 30})
    assert var_5 is True
    var_6 = '_precord_buckets'
    var_7 = bool('_precord_buckets' not in {'name': 'John', 'age': 30})
    assert var_7 is True



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_multiple_kwargs. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = var_7._size
    var_9 = var_7._buckets

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = module_0.field(initial=var_1)
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda : [1, 2, 3]
    var_2 = module_0.field(initial=var_1)
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 15
    var_4 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = 20
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 999
    var_3 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = '5'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 5/12 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/9 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/8 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/11 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 6/11 statements.
# Partially parsed test_precord_new_empty. Retrieved 4/9 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = {}
    var_3 = ()
    var_4 = 0

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = {}
    var_3 = ()
    var_4 = 'test'

def test_case_0():
    var_0 = 'value'
    var_1 = set()
    var_2 = 'value'
    var_3 = lambda : 100
    var_4 = {var_2: var_3}
    var_5 = ()

def test_case_0():
    var_0 = 'num'
    var_1 = set()
    var_2 = {}
    var_3 = ()
    var_4 = 'num'
    var_5 = {var_4}
    var_6 = '5'

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = {}
    var_3 = ()
    var_4 = True
    var_5 = 'test'
    var_6 = 'ignored'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = {}
    var_3 = ()



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 3/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 0



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_partial_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_overrides_initial. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_internal_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Failed to parse test_precord_constructor_empty.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100
    var_5 = 200

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 42
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999
    var_4 = 'extra_field'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_result_when_not_dirty_and_correct_type. Retrieved 4/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 5/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/19 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/21 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 4/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockPRecord'
    var_5 = module_0.pmap()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'MockPRecord.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockPRecord'
    var_4 = module_0.pmap()
    var_5 = 'error_code_1'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'MockPRecord'
    var_4 = module_0.pmap()
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(var_0)
    var_7 = bool(len(var_0) > 0)
    assert var_7 is True
    var_8 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 8/16 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 8/16 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 8/17 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 5/12 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 6/17 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 8/21 statements.
# Partially parsed test_precord_meta_new_removes_pfield_from_dct. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = 'TestRecord'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = 'TestRecord'
    var_7 = '_precord_mandatory_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default_value'
    var_6 = 'TestRecord'
    var_7 = '_precord_initial_values'
    var_8 = 'field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = 'TestRecord'
    var_4 = '__slots__'

def test_case_0():
    var_0 = 'field1'
    var_1 = '__invariant__'
    var_2 = True
    var_3 = None
    var_4 = 'TestRecord'
    var_5 = '_precord_invariants'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = '_precord_fields'
    var_2 = True
    var_3 = None
    var_4 = 'parent_field'
    var_5 = 'child_field'
    var_6 = False
    var_7 = 'child_default'
    var_8 = 'ChildRecord'
    var_9 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = 'TestRecord'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 4/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 4/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 5/17 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = module_1.pmap(var_1)
    var_3 = 'error_code_1'
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = module_1.pmap(var_1)
    var_3 = 'TestRecord.missing_field'
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = module_1.pmap(var_1)
    var_3 = 'error_1'
    var_4 = 'TestRecord.field1'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_value. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_override_initial_values. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = var_7._size
    var_9 = var_7._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda : 42
    var_2 = module_0.field(initial=var_1)
    var_3 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 10
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_present. Retrieved 6/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 6/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields. Retrieved 7/19 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'error_code_1'
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'TestRecord.y'
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'error_code_1'
    var_6 = 'TestRecord.y'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 13/26 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'default'
    var_3 = False
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = '_precord_mandatory_fields'
    var_12 = 'field1'
    var_13 = 'field2'
    var_14 = '_precord_initial_values'
    var_15 = 'field2'
    var_16 = 'field1'
    var_17 = '__slots__'
    var_18 = '_precord_invariants'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 8/15 statements.
# Partially parsed test_set_with_invalid_field_raises_attribute_error. Retrieved 8/14 statements.
# Partially parsed test_set_with_setitem. Retrieved 8/15 statements.
# Partially parsed test_set_with_factory_fields_restriction. Retrieved 16/28 statements.
# Partially parsed test_set_with_failed_invariant. Retrieved 8/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = 'name'
    var_5 = 'test'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 'new_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'invalid_field'
    var_7 = 'value'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "'invalid_field' is not among the specified fields"

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = 'age'
    var_5 = 25
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 30

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = (var_0, var_1)
    var_5 = lambda x: var_4
    var_6 = 'name'
    var_7 = 'age'
    var_8 = set()
    var_9 = []
    var_10 = 'TestClass'
    var_11 = 'name'
    var_12 = 'age'
    var_13 = 'test'
    var_14 = 25
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = module_0.pmap(var_15)
    var_17 = 'updated'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'value'
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = 'value'
    var_5 = 10
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 20
    var_9 = 'error_code'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 7/14 statements.
# Partially parsed test_set_with_invalid_field_name. Retrieved 8/14 statements.
# Partially parsed test_set_with_factory_fields_filter. Retrieved 7/16 statements.
# Partially parsed test_set_with_factory_fields_excluded. Retrieved 8/16 statements.
# Partially parsed test_setitem_delegates_to_set. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'name'
    var_7 = 'test_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'invalid_field'
    var_7 = 'value'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'invalid_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestClass'
    var_4 = 'field1'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = 'test_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestClass'
    var_4 = 'field1'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = []
    var_8 = 'test_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'name'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_precord_initial_values_predicate. Retrieved 8/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = lambda : 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'x'
    var_8 = 'y'
    var_9 = 10
    var_10 = lambda : var_9
    var_11 = 20



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_predicate_line_6_evaluates_to_true_when_is_dirty. Retrieved 6/13 statements.
# Partially parsed test_predicate_line_6_evaluates_to_true_when_not_isinstance. Retrieved 5/15 statements.
# Partially parsed test_predicate_line_6_evaluates_to_true_when_both_conditions_met. Retrieved 9/17 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 2

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = module_1.pmap(var_4)
    var_6 = 5
    var_7 = 'y'
    var_8 = 10



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'name'
    var_2 = 'new_value'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 5/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 5/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields_present. Retrieved 6/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = 'name'
    var_4 = module_0.field()
    var_5 = {var_3: var_4}
    var_6 = 'error_code_1'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = 'name'
    var_4 = module_0.field()
    var_5 = {var_3: var_4}
    var_6 = 'TestRecord.name'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = 'name'
    var_4 = module_0.field()
    var_5 = {var_3: var_4}
    var_6 = 'error_code_1'
    var_7 = 'TestRecord.name'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_multiple_fields_with_special_values. Retrieved 6/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'TestRecord'
    var_5 = 'name='
    var_6 = "'John'"
    var_7 = 'age='
    var_8 = '30'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'hello'
    var_4 = 42
    var_5 = True
    var_6 = 'ComplexRecord'
    var_7 = "text='hello'"
    var_8 = 'number=42'
    var_9 = 'flag=True'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 5/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 5/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 7/20 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockCls'
    var_5 = module_0.pmap()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'MockCls.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = module_0.pmap()
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error1'
    var_9 = 'error2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockCls'
    var_3 = module_0.pmap()
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_precord_initial_values_predicate. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 10
    var_5 = lambda : 20
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_present. Retrieved 6/24 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = None
    var_5 = False



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_precord_meta_new_returns_class. Retrieved 7/14 statements.


def test_case_0():
    var_0 = True
    var_1 = '_precord_fields'
    var_2 = '__invariant__'
    var_3 = 'test_field'
    var_4 = None
    var_5 = ()
    var_6 = 'TestPRecord'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_evolver_persistent_predicate_is_dirty_true. Retrieved 3/15 statements.
# Partially parsed test_precord_evolver_persistent_predicate_not_isinstance_true. Retrieved 1/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'x'
    var_3 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 4/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 5/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 6/19 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 3/18 statements.
# Partially parsed test_persistent_with_clean_state_returns_original_pmap_if_same_type. Retrieved 4/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockClass'
    var_5 = module_0.pmap()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'MockClass.field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()
    var_4 = 'error1'
    var_5 = 'error2'
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = module_0.pmap()
    var_4 = len(var_0)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = module_0.pmap()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 5/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present. Retrieved 8/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = []
    var_5 = 'error_code_1'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Field invariant failed'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = []
    var_5 = 'MockCls.field1'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Field invariant failed'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = []
    var_5 = 'error_code_1'
    var_6 = 'error_code_2'
    var_7 = 'MockCls.field1'
    var_8 = 'MockCls.field2'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'Field invariant failed'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_precord_evolver_set_with_field_found. Retrieved 7/13 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = 'Alice'
    var_3 = 30
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.pmap(var_4)
    var_6 = 'Bob'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_internal_buckets. Retrieved 7/10 statements.
# Failed to parse test_precord_constructor_empty.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 8/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._PMap__size
    var_6 = var_4._PMap__buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = 4



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_precord_new_predicate_false_missing_precord_size. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_missing_precord_buckets. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_missing_both. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 30



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serialize_with_no_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 2/11 statements.
# Partially parsed test_serialize_mixed_fields. Retrieved 5/12 statements.
# Partially parsed test_serialize_empty_record. Retrieved 2/6 statements.
# Partially parsed test_serialize_preserves_types. Retrieved 9/13 statements.
# Partially parsed test_serialize_with_none_serializer. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'
    var_2 = 30

def test_case_0():
    var_0 = 'Active'
    var_1 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'john'
    var_4 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(serializer=var_0)
    var_2 = 42



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values_override. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values_callable. Retrieved 1/7 statements.
# Failed to parse test_precord_new_empty_record.
# Partially parsed test_precord_new_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_precord_new_with_kwargs_and_initial_values. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.field()
    var_6 = module_0.field()
    var_7 = module_0.field()
    var_8 = 30
    var_9 = 25



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_persistent_with_mandatory_fields. Retrieved 4/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_precord_meta_new_sets_precord_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 10/18 statements.
# Partially parsed test_precord_meta_new_sets_empty_slots. Retrieved 5/11 statements.
# Partially parsed test_precord_meta_new_sets_invariants. Retrieved 6/16 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 8/17 statements.
# Partially parsed test_precord_meta_new_no_fields. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 42
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 42
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_mandatory_fields'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = True
    var_4 = False
    var_5 = 42
    var_6 = 'default'
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_initial_values'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = ()
    var_3 = 'TestRecord'
    var_4 = '__slots__'

def test_case_0():
    var_0 = 'field1'
    var_1 = '__invariant__'
    var_2 = True
    var_3 = ()
    var_4 = 'TestRecord'
    var_5 = '_precord_invariants'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = True
    var_2 = ()
    var_3 = 'Parent'
    var_4 = 'child_field'
    var_5 = False
    var_6 = 10
    var_7 = 'Child'
    var_8 = 'parent_field'
    var_9 = 'child_field'

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'EmptyRecord'
    var_3 = [var_2, var_1, var_0]
    var_4 = '_precord_fields'
    var_5 = set()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 3/11 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields. Retrieved 6/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'john'
    var_2 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42
    var_2 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 12/23 statements.


def test_case_0():
    var_0 = 'default'
    var_1 = True
    var_2 = False
    var_3 = 'test_attr'
    var_4 = 'optional_attr'
    var_5 = 'TestRecord'
    var_6 = ()
    var_7 = '_precord_fields'
    var_8 = '_precord_invariants'
    var_9 = '_precord_mandatory_fields'
    var_10 = '_precord_initial_values'
    var_11 = '__slots__'
    var_12 = 'test_attr'
    var_13 = 'optional_attr'
    var_14 = 'test_attr'
    var_15 = 'optional_attr'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_partial_kwargs. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = lambda : 100
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 99
    var_5 = 88

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 3



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_precord_new_predicate_false_when_only_precord_size_in_kwargs. Retrieved 2/5 statements.
# Partially parsed test_precord_new_predicate_false_when_only_precord_buckets_in_kwargs. Retrieved 2/5 statements.
# Partially parsed test_precord_new_predicate_false_when_neither_special_kwargs_present. Retrieved 4/7 statements.
# Partially parsed test_precord_new_predicate_false_with_regular_field_only. Retrieved 2/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_repr_with_special_characters. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord'
    var_5 = 'name='
    var_6 = "'Alice'"
    var_7 = 'age='
    var_8 = '30'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 'two'
    var_5 = 3.0
    var_6 = 'MultiFieldRecord'
    var_7 = 'first=1'
    var_8 = "second='two'"
    var_9 = 'third=3.0'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = "hello'world"
    var_2 = 'SpecialRecord'
    var_3 = 'text='
    var_4 = "hello'world"



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_empty_record.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : []
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 999
    var_3 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'name'
    var_5 = 'age'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_present. Retrieved 6/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = []
    var_5 = 1
    var_6 = 2



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 13/24 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 42
    var_3 = False
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'TestClass'
    var_7 = ()
    var_8 = '_precord_fields'
    var_9 = '_precord_invariants'
    var_10 = '_precord_mandatory_fields'
    var_11 = '_precord_initial_values'
    var_12 = '__slots__'
    var_13 = 'field1'
    var_14 = 'field2'
    var_15 = 'field2'
    var_16 = 'field1'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_precord_new_predicate_false_missing_precord_size. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_missing_precord_buckets. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_missing_both. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 7/20 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 6/19 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 7/20 statements.
# Partially parsed test_precord_meta_new_sets_empty_slots. Retrieved 2/13 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 6/4 statements.
# Partially parsed test_precord_meta_new_removes_field_descriptors_from_dct. Retrieved 4/15 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 6/21 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = True
    var_4 = False
    var_5 = 'default_value'
    var_6 = 'TestRecord'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = 'field3'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = True
    var_4 = False
    var_5 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = False
    var_4 = 'value1'
    var_5 = 42
    var_6 = 'TestRecord'
    var_7 = 'field3'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'TestRecord'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = 'TestRecord'
    var_6 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = 'TestRecord'
    var_6 = '_precord_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'regular_attr'
    var_2 = 'some_value'
    var_3 = 'TestRecord'
    var_4 = 'field1'
    var_5 = 'regular_attr'

def test_case_0():
    var_0 = 'base_field'
    var_1 = True
    var_2 = 'BaseRecord'
    var_3 = 'child_field'
    var_4 = False
    var_5 = 'ChildRecord'
    var_6 = 'base_field'
    var_7 = 'child_field'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_precord_new_predicate_false_when_missing_precord_size. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_when_missing_precord_buckets. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_when_both_missing. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5



