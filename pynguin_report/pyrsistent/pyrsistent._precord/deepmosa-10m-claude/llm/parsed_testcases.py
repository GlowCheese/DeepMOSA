####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_precord_when_not_dirty. Retrieved 4/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_error_codes. Retrieved 5/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_fields. Retrieved 5/18 statements.
# Partially parsed test_persistent_checks_mandatory_fields. Retrieved 5/17 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 3/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()
    var_4 = 'error1'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'error1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.pmap()
    var_4 = 'field1'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestRecord'
    var_1 = {}
    var_2 = 'required_field'
    var_3 = {var_2}
    var_4 = []
    var_5 = module_0.pmap()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestRecord.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = module_0.pmap()
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'global_error'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_with_factory_fields_dict. Retrieved 3/6 statements.


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
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = 'John'
    var_4 = 'value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = 'extra'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = 'Test'

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'Alice'
    var_4 = 28
    var_5 = 'alice@example.com'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = 'Bob'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values_and_kwargs_override. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.


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
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 5
    var_5 = 10
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = lambda : 42
    var_3 = {var_1: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = set()
    var_3 = 5
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 5



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 3/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 3/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_present. Retrieved 5/14 statements.


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



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_precord_initial_values_predicate. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'default_name'
    var_5 = lambda : 0
    var_6 = {var_2: var_4, var_3: var_5}



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 7/17 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 10/19 statements.
# Partially parsed test_precord_meta_new_sets_empty_slots. Retrieved 6/12 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 6/17 statements.
# Partially parsed test_precord_meta_new_returns_type_instance. Retrieved 5/11 statements.


def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 42
    var_3 = True
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = '__invariant__'
    var_4 = ()
    var_5 = 'TestRecord'
    var_6 = '_precord_invariants'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = True
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_mandatory_fields'
    var_8 = 'field2'
    var_9 = 'field1'

def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 42
    var_3 = 'test'
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'field3'
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_initial_values'
    var_10 = 'field1'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = ()
    var_4 = 'TestRecord'
    var_5 = '__slots__'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = '_precord_fields'
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'DerivedRecord'
    var_7 = 'field1'
    var_8 = 'field2'

def test_case_0():
    var_0 = None
    var_1 = False
    var_2 = 'field1'
    var_3 = ()
    var_4 = 'TestRecord'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_overrides_initial_values. Retrieved 5/8 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.


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
    var_1 = set()
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 7

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = lambda : 200
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'z'
    var_1 = 100
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = 5
    var_6 = 300

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 13/26 statements.


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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_persistent_predicate_is_dirty_true. Retrieved 6/13 statements.
# Partially parsed test_persistent_predicate_not_isinstance_true. Retrieved 7/14 statements.
# Partially parsed test_persistent_predicate_both_conditions_true. Retrieved 9/17 statements.


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
    var_5 = {var_1: var_2}
    var_6 = module_1.pmap(var_5)

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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty_and_already_correct_type. Retrieved 12/24 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 12/25 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_error. Retrieved 13/27 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 11/27 statements.
# Partially parsed test_persistent_with_dirty_state_creates_new_instance. Retrieved 14/31 statements.


def test_case_0():
    var_0 = 'MockField'
    var_1 = 'factory'
    var_2 = 'invariant'
    var_3 = [var_1, var_2]
    var_4 = lambda x: x
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'test_field'
    var_10 = set()
    var_11 = []
    var_12 = 'MockPRecord'
    var_13 = []

def test_case_0():
    var_0 = 'MockField'
    var_1 = 'factory'
    var_2 = 'invariant'
    var_3 = [var_1, var_2]
    var_4 = lambda x: x
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'test_field'
    var_10 = 'required_field'
    var_11 = {var_10}
    var_12 = []
    var_13 = 'MockPRecord'
    var_14 = []
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'MockPRecord.required_field'

def test_case_0():
    var_0 = 'MockField'
    var_1 = 'factory'
    var_2 = 'invariant'
    var_3 = [var_1, var_2]
    var_4 = lambda x: x
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'test_field'
    var_10 = set()
    var_11 = []
    var_12 = 'MockPRecord'
    var_13 = []
    var_14 = 'error_code_1'
    var_15 = bool(False)
    assert var_15 is True
    var_16 = 'error_code_1'

def test_case_0():
    var_0 = 'MockField'
    var_1 = 'factory'
    var_2 = 'invariant'
    var_3 = [var_1, var_2]
    var_4 = lambda x: x
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'test_field'
    var_10 = set()
    var_11 = 'MockPRecord'
    var_12 = []
    var_13 = bool(False)
    assert var_13 is True
    var_14 = 'global_error'

def test_case_0():
    var_0 = 'MockField'
    var_1 = 'factory'
    var_2 = 'invariant'
    var_3 = [var_1, var_2]
    var_4 = lambda x: x
    var_5 = True
    var_6 = None
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = 'test_field'
    var_10 = set()
    var_11 = []
    var_12 = 'MockPRecord'
    var_13 = []
    var_14 = 'test_field'
    var_15 = 'value'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 3/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/20 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 3/19 statements.
# Partially parsed test_persistent_with_dirty_state. Retrieved 3/16 statements.


def test_case_0():
    var_0 = set()
    var_1 = []
    var_2 = {}
    var_3 = []

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = []
    var_4 = {}
    var_5 = 'MockCls'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = set()
    var_1 = []
    var_2 = {}
    var_3 = 'MockCls'
    var_4 = []
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error1'
    var_9 = 'error2'

def test_case_0():
    var_0 = set()
    var_1 = {}
    var_2 = 'MockCls'
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

def test_case_0():
    var_0 = set()
    var_1 = []
    var_2 = {}
    var_3 = []



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 2/10 statements.
# Partially parsed test_precord_evolver_set_with_invalid_field_name. Retrieved 2/10 statements.
# Partially parsed test_precord_evolver_setitem. Retrieved 2/9 statements.
# Partially parsed test_precord_evolver_set_with_type_check. Retrieved 2/11 statements.
# Partially parsed test_precord_evolver_set_with_factory_fields. Retrieved 2/12 statements.
# Partially parsed test_precord_evolver_set_with_factory_fields_excluded. Retrieved 3/11 statements.


def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = 42

def test_case_0():
    var_0 = []
    var_1 = 'nonexistent'
    var_2 = 42
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'nonexistent'

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = 99

def test_case_0():
    var_0 = []
    var_1 = 'x'
    var_2 = 'not_an_int'
    var_3 = bool(False)
    assert var_3 is True

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = 42

def test_case_0():
    var_0 = 'x'
    var_1 = []
    var_2 = set()
    var_3 = 42



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_precord_repr. Retrieved 6/12 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_multiple_fields. Retrieved 8/12 statements.
# Partially parsed test_precord_repr_with_none. Retrieved 2/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord('
    var_5 = 'name='
    var_6 = "'Alice'"
    var_7 = 'age='
    var_8 = '30'
    var_9 = ')'

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
    var_4 = 'text'
    var_5 = 2
    var_6 = 3
    var_7 = [var_3, var_5, var_6]
    var_8 = 'MultiFieldRecord('
    var_9 = 'a=1'
    var_10 = "b='text'"
    var_11 = 'c=[1, 2, 3]'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = 'data=None'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 9/26 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 9/27 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 10/29 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 8/29 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = 0
    var_5 = 'MockClass'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_1}
    var_3 = []
    var_4 = None
    var_5 = 0
    var_6 = 'MockClass'
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'MockClass.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = 0
    var_5 = 'MockClass'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = 'field_error'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'field_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = None
    var_3 = 0
    var_4 = 'MockClass'
    var_5 = 'field1'
    var_6 = 'value1'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'global_error'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_persistent_predicate_is_dirty_true. Retrieved 6/13 statements.
# Partially parsed test_persistent_predicate_not_isinstance_true. Retrieved 5/14 statements.
# Partially parsed test_persistent_predicate_both_conditions. Retrieved 7/14 statements.


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
    var_5 = [var_4]

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 1
    var_4 = {var_2: var_3}
    var_5 = module_1.pmap(var_4)
    var_6 = 2



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_set_with_valid_field. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'name'
    var_5 = 'Jane'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 5/10 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_invariant_error_codes. Retrieved 6/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 5/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_both_errors_and_missing. Retrieved 6/14 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = module_0.pmap()
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
    var_4 = 'TestClass'
    var_5 = module_0.pmap()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestClass.field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'TestClass'
    var_5 = module_0.pmap()
    var_6 = 'error1'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error1'
    var_9 = 'TestClass.field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'TestClass'
    var_3 = module_0.pmap()
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 4/18 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 5/11 statements.
# Partially parsed test_set_with_type_check_failure. Retrieved 4/19 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 4/17 statements.
# Partially parsed test_set_with_factory_fields_filter. Retrieved 4/19 statements.
# Partially parsed test_setitem_delegates_to_set. Retrieved 4/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'MockClass'
    var_2 = module_0.pmap()
    var_3 = 'test_field'
    var_4 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'MockClass'
    var_2 = module_0.pmap()
    var_3 = 'nonexistent_field'
    var_4 = 42
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'nonexistent_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'MockClass'
    var_2 = module_0.pmap()
    var_3 = 'test_field'
    var_4 = 'not_an_int'
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'MockClass'
    var_2 = module_0.pmap()
    var_3 = 'test_field'
    var_4 = 42
    var_5 = 'test_error_code'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'MockClass'
    var_2 = 'test_field'
    var_3 = module_0.pmap()
    var_4 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'test_field'
    var_1 = 'MockClass'
    var_2 = module_0.pmap()
    var_3 = 'test_field'
    var_4 = 42



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 12/22 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 42
    var_3 = False
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'TestPRecord'
    var_7 = ()
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = '_precord_mandatory_fields'
    var_12 = '_precord_initial_values'
    var_13 = '_precord_invariants'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_format. Retrieved 5/9 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_with_nested_values. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 4/8 statements.
# Partially parsed test_serialize_multiple_fields. Retrieved 8/12 statements.


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
    var_2 = 'Bob'
    var_3 = 25
    var_4 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Charlie'
    var_3 = 'nested'
    var_4 = 'value'
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Dave'
    var_3 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = 1
    var_5 = 'two'
    var_6 = 3.0
    var_7 = True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 3/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/20 statements.
# Partially parsed test_persistent_calls_global_invariants. Retrieved 4/21 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 4/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = 'required_field'
    var_3 = {var_2}
    var_4 = []
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'MockCls.required_field'

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = []
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error1'
    var_9 = 'error2'

def test_case_0():
    var_0 = []
    var_1 = 'MockCls'
    var_2 = {}
    var_3 = set()
    var_4 = []
    var_5 = bool(False)
    assert var_5 is True
    var_6 = len(var_0)
    var_7 = bool(len(var_0) > 0)
    assert var_7 is True
    var_8 = 'global_error'

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = []



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 6/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 6/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 9/18 statements.
# Partially parsed test_persistent_predicate_false_when_no_errors_or_missing_fields. Retrieved 5/13 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'error1'
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
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = 'TestRecord.y'
    var_8 = 'TestRecord.z'
    var_9 = bool(False)
    assert var_9 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_precord_initial_values_predicate. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'default_name'
    var_5 = lambda : 0
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 25



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_successfully. Retrieved 10/20 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'test_field'
    var_3 = ()
    var_4 = 'TestPRecord'
    var_5 = '_precord_fields'
    var_6 = '_precord_invariants'
    var_7 = '_precord_mandatory_fields'
    var_8 = '_precord_initial_values'
    var_9 = '__slots__'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 6/11 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = False



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_precord_new_predicate_false_when_missing_precord_size. Retrieved 2/5 statements.
# Partially parsed test_precord_new_predicate_false_when_missing_precord_buckets. Retrieved 2/5 statements.
# Partially parsed test_precord_new_predicate_false_when_both_missing. Retrieved 2/5 statements.


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
    var_1 = 'test'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 1/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30



# Parsed testcases at query #29
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



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 6/10 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values_and_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 2/5 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 2
    var_3 = module_1.pmap(pre_size=var_2)
    var_4 = var_3._buckets
    var_5 = 0

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
    var_2 = 'x'
    var_3 = 10
    var_4 = {var_2: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = lambda : 42
    var_3 = {var_1: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 2/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 2/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields. Retrieved 4/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'error1'
    var_3 = bool(False)
    assert var_3 is True

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
    var_2 = 'error1'
    var_3 = 'error2'
    var_4 = 'TestRecord.name'
    var_5 = bool(False)
    assert var_5 is True



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 4/13 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_with_serializers. Retrieved 4/14 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 3/10 statements.


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
    var_1 = 'hello'
    var_2 = 'world'
    var_3 = 'uppercase'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'hello'
    var_3 = 'unchanged'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = 'value'



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_precord_repr. Retrieved 6/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord('
    var_5 = ')'
    var_6 = 'name='
    var_7 = "'Alice'"
    var_8 = 'age='
    var_9 = '30'



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_sets_empty_slots. Retrieved 4/9 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 9/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 8/17 statements.
# Partially parsed test_precord_meta_new_returns_type_instance. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 'default_value'
    var_5 = ()
    var_6 = 'TestPRecord'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = True
    var_4 = False
    var_5 = 'default_value'
    var_6 = ()
    var_7 = 'TestPRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = True
    var_4 = False
    var_5 = 'default_value'
    var_6 = 42
    var_7 = ()
    var_8 = 'TestPRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = ()
    var_3 = 'TestPRecord'

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = ()
    var_7 = 'TestPRecord'
    var_8 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = 'valid'
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = ()
    var_7 = 'TestPRecord'
    var_8 = '_precord_invariants'

def test_case_0():
    var_0 = 'base_field'
    var_1 = True
    var_2 = 'BaseRecord'
    var_3 = ()
    var_4 = 'child_field'
    var_5 = False
    var_6 = 'child_default'
    var_7 = 'ChildRecord'
    var_8 = 'base_field'
    var_9 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = ()
    var_3 = 'TestPRecord'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 13/26 statements.


def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 42
    var_3 = False
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'TestClass'
    var_7 = ()
    var_8 = '__slots__'
    var_9 = '_precord_fields'
    var_10 = 'field1'
    var_11 = 'field2'
    var_12 = '_precord_mandatory_fields'
    var_13 = '_precord_initial_values'
    var_14 = '_precord_invariants'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 7/20 statements.
# Partially parsed test_persistent_raises_on_missing_mandatory_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_raises_on_invariant_error_codes. Retrieved 8/21 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 7/18 statements.


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
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockCls.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = 'error2'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = {}
    var_2 = set()
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 12/25 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = False
    var_3 = 42
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'TestClass'
    var_7 = '_precord_fields'
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = '_precord_mandatory_fields'
    var_11 = 'field1'
    var_12 = 'field2'
    var_13 = '_precord_initial_values'
    var_14 = 'field1'
    var_15 = 'field2'
    var_16 = '__slots__'
    var_17 = '_precord_invariants'



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_special_precord_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.


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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._precord_size
    var_6 = var_4._precord_buckets

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
    var_3 = 999
    var_4 = 'extra_field'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 10
    var_4 = 'x'
    var_5 = 'y'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 4/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 6/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_invariant_errors. Retrieved 7/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 8/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'TestRecord'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TestRecord.field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'TestRecord'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestRecord'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_exist. Retrieved 2/11 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_exist. Retrieved 2/11 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_exist. Retrieved 4/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error1'
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
    var_1 = 'error1'
    var_2 = 'error2'
    var_3 = 'TestRecord.name'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_internal_buckets. Retrieved 10/13 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.


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
    var_1 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_precord_meta_new_basic. Retrieved 10/21 statements.
# Partially parsed test_precord_meta_new_mandatory_fields. Retrieved 6/14 statements.
# Partially parsed test_precord_meta_new_initial_values. Retrieved 8/16 statements.
# Partially parsed test_precord_meta_new_with_invariant. Retrieved 8/4 statements.
# Partially parsed test_precord_meta_new_slots. Retrieved 3/9 statements.
# Partially parsed test_precord_meta_new_empty_fields. Retrieved 3/6 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 'default_value'
    var_5 = 'TestRecord'
    var_6 = '_precord_fields'
    var_7 = '_precord_invariants'
    var_8 = '_precord_mandatory_fields'
    var_9 = '_precord_initial_values'

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
    var_3 = True
    var_4 = False
    var_5 = 'default_value'
    var_6 = 42
    var_7 = 'TestRecord'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = 'TestRecord'
    var_7 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = 'TestRecord'
    var_7 = '_precord_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 'TestRecord'

def test_case_0():
    var_0 = {}
    var_1 = 'TestRecord'
    var_2 = set()



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_or_missing_fields. Retrieved 5/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields. Retrieved 5/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields. Retrieved 6/19 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'error_code_1'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'TestRecord.name'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'Field invariant failed'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'error_code_1'
    var_5 = 'TestRecord.name'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'Field invariant failed'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 4/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 4/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 5/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'error_code_1'
    var_3 = False
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'TestRecord.x'
    var_3 = False
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'error_code_1'
    var_3 = 'TestRecord.x'
    var_4 = False
    var_5 = True
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 5/13 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
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
    var_1 = 42
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = set()
    var_3 = 100
    var_4 = 200

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_set_field_exists. Retrieved 8/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'processed_value'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = 'test_key'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = 'test_value'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'name'
    var_5 = 'age'



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_precord_repr. Retrieved 5/11 statements.


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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_special_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_partial_override. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.


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
    var_1 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999
    var_4 = 'extra_field'

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100
    var_5 = 200

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
    var_1 = 1
    var_2 = 'x'
    var_3 = {var_2}



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_persistent_predicate_line_6_true_when_dirty. Retrieved 3/11 statements.
# Partially parsed test_persistent_predicate_line_6_true_when_not_isinstance. Retrieved 4/14 statements.
# Partially parsed test_persistent_predicate_line_11_true_with_mandatory_fields. Retrieved 2/11 statements.
# Partially parsed test_persistent_predicate_line_15_true_with_invariant_errors. Retrieved 2/13 statements.
# Partially parsed test_persistent_predicate_line_15_true_with_missing_fields. Retrieved 2/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'x'
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = []
    var_3 = 'x'
    var_4 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'error1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'TestRecord.x'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_precord_buckets. Retrieved 3/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 0



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra_false. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_empty.
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
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = None

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



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_persistent_predicate_line_6_true_when_dirty. Retrieved 6/16 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 2



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 5/14 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_defaults. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_defaults_and_kwargs. Retrieved 4/7 statements.
# Failed to parse test_precord_constructor_with_callable_default.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_partial_kwargs. Retrieved 5/8 statements.


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
    var_2 = 'name'
    var_3 = 'age'

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
    var_1 = None
    var_2 = 'Test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'Test'
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'John'
    var_4 = 'john@example.com'
    var_5 = 'age'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
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
    var_2 = 'x'
    var_3 = {var_2}

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



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_override. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_partial_fields. Retrieved 3/6 statements.


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
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 1
    var_6 = var_4._pmap_buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 'y'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 7/10 statements.


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

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)

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



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_partial_fields. Retrieved 5/8 statements.


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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = 100
    var_5 = 200

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = True
    var_4 = 'z'

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
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 3
    var_5 = 'y'



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.


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
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = None



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_initial_values_callable.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
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
    var_4 = 'Jane'
    var_5 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'
    var_2 = 'ignored'
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



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 5/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 7/20 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 8/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = module_0.pmap(var_3)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}
    var_2 = 'required_field'
    var_3 = {var_2}
    var_4 = []
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TestClass.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error_code_1'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}
    var_2 = set()
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = '_is_new'



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_partial_kwargs. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_with_callable_initial.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_internal_creation. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.


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
    var_4 = 'Alice'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'John'
    var_3 = 'ignored'
    var_4 = 'extra_field'

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Default'
    var_1 = module_0.field(initial=var_0)
    var_2 = 'name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'DefaultName'
    var_1 = module_0.field(initial=var_0)
    var_2 = 25
    var_3 = module_0.field(initial=var_2)
    var_4 = 'Bob'
    var_5 = 35

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = 'Test'



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_precord_evolver_persistent_predicate_line_6. Retrieved 9/20 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)
    var_8 = 'Jane'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_value. Retrieved 2/8 statements.
# Partially parsed test_precord_new_kwargs_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.


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
    var_1 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 100
    var_1 = module_0.field(initial=var_0)
    var_2 = 200
    var_3 = module_0.field(initial=var_2)
    var_4 = 1
    var_5 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_special_attributes. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'active'
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Bob'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Charlie'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = True
    var_3 = 'David'
    var_4 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'Eve'
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'Frank'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/5 statements.
# Failed to parse test_precord_constructor_empty.
# Partially parsed test_precord_constructor_with_internal_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.


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
    var_3 = True
    var_4 = 'y'



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/15 statements.


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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 6/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 7/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_with_clean_pmap_returns_original. Retrieved 6/19 statements.
# Partially parsed test_persistent_collects_multiple_invariant_error_codes. Retrieved 9/22 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'TestClass'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TestClass.field1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = 'error2'
    var_8 = 'error3'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'error1'
    var_11 = 'error2'
    var_12 = 'error3'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 3/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 0



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_internal_creation. Retrieved 7/10 statements.


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
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'

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
    var_1 = module_0.field()
    var_2 = True
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 999
    var_3 = 'z'

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



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_repr_with_special_characters. Retrieved 3/8 statements.


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
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'MultiFieldRecord'
    var_7 = 'x=1'
    var_8 = 'y=2'
    var_9 = 'z=3'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello\nworld'
    var_2 = 'SpecialRecord'
    var_3 = 'text='
    var_4 = "'"



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/23 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/17 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 4/17 statements.
# Partially parsed test_persistent_converts_pmap_to_destination_class. Retrieved 5/20 statements.


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
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'TestClass'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = 'MockClass'
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []
    var_5 = '_is_mock'



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_persistent_predicate_is_dirty_true. Retrieved 9/15 statements.
# Partially parsed test_persistent_predicate_not_isinstance_true. Retrieved 6/21 statements.
# Partially parsed test_persistent_predicate_both_conditions_true. Retrieved 7/22 statements.


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
    var_8 = 'updated'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'name'
    var_6 = bool(var_5)
    assert var_6 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'test'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'modified'
    var_6 = 'name'
    var_7 = bool(var_6)
    assert var_7 is True



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_persistent_with_no_errors. Retrieved 6/23 statements.
# Partially parsed test_persistent_with_invariant_error_codes. Retrieved 7/20 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 5/20 statements.
# Partially parsed test_persistent_returns_same_instance_when_not_dirty. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'TestClass'
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 2/11 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_with_serializers. Retrieved 4/14 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'alice'
    var_2 = 30

def test_case_0():
    var_0 = 42
    var_1 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'test'
    var_3 = 'a record'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = None



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 12/24 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = False
    var_3 = 'default_value'
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = 'TestClass'
    var_7 = ()
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'
    var_11 = '_precord_mandatory_fields'
    var_12 = 'field1'
    var_13 = 'field2'
    var_14 = '_precord_initial_values'
    var_15 = 'field1'
    var_16 = 'field2'
    var_17 = '_precord_invariants'



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_precord_evolver_set_with_field_found. Retrieved 17/27 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = 'factory'
    var_2 = 'type'
    var_3 = 'invariant'
    var_4 = [var_1, var_2, var_3]
    var_5 = lambda x: x
    var_6 = ()
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'MockClass'
    var_12 = 'test_key'
    var_13 = set()
    var_14 = []
    var_15 = module_0.pmap()
    var_16 = 'test_key'
    var_17 = 'test_value'



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 3/8 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_special_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_overwrites_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 25
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Jane'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test1'
    var_2 = 'test2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'John'
    var_3 = 'ignored'
    var_4 = 'extra'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'John'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._precord_size
    var_6 = var_4._precord_buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Default'
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 'Custom'
    var_4 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_precord_meta_new_returns_class. Retrieved 11/20 statements.


def test_case_0():
    var_0 = True
    var_1 = '_precord_fields'
    var_2 = '__invariant__'
    var_3 = {}
    var_4 = None
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = ()
    var_7 = 'TestPRecord'
    var_8 = '_precord_invariants'
    var_9 = '_precord_mandatory_fields'
    var_10 = '_precord_initial_values'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_initial_values_override. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_callable_initial_value. Retrieved 3/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 42
    var_3 = {var_1: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 42
    var_3 = {var_1: var_2}
    var_4 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = module_0.field()
    var_3 = 'x'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_persistent_returns_dirty_precord_instance. Retrieved 8/20 statements.
# Partially parsed test_persistent_returns_clean_precord_when_not_dirty. Retrieved 8/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 8/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_error. Retrieved 9/23 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 7/23 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestRecord'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestRecord'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'TestRecord'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'TestRecord.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestRecord'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)
    var_8 = 'error_code_1'
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'TestRecord'
    var_3 = 'key'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'global_error'



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
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
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 15

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = {}
    var_3 = 5
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 5
    var_3 = 99
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = 99

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_and_override. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.


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



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_persistent_returns_same_instance_when_not_dirty. Retrieved 8/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 8/16 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 7/16 statements.
# Partially parsed test_persistent_raises_exception_with_both_missing_fields_and_error_codes. Retrieved 7/15 statements.


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
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockClass'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockClass.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error_code_1'
    var_7 = 'error_code_2'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = len(var_0)
    var_7 = bool(var_6 > 0)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockClass'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = 'error_code_1'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'MockClass.required_field'



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 4/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'
    var_2 = 'name'
    var_3 = 'modified'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 5/9 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.


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
    var_1 = 1
    var_2 = {}
    var_3 = {}
    var_4 = (var_2, var_3)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = 'x'
    var_3 = [var_2]

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



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 25
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Jane'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 'Alice'
    var_4 = 28

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'Bob'
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'active'
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Charlie'
    var_4 = 'inactive'



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 4/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_precord_new_predicate_false_missing_precord_size. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_missing_precord_buckets. Retrieved 2/6 statements.
# Partially parsed test_precord_new_predicate_false_neither_special_attributes. Retrieved 2/6 statements.


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
    var_1 = 42



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 4/10 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
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
    var_2 = 10
    var_3 = 20

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
    var_1 = 100
    var_2 = 999
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 42
    var_2 = {var_0: var_1}
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = lambda : 99
    var_2 = {var_0: var_1}
    var_3 = module_0.field()

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



# Parsed testcases at query #98
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 6/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 6/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields. Retrieved 8/20 statements.


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
    var_5 = 'missing_field_1'
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
    var_6 = 'error_code_2'
    var_7 = 'missing_field_1'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #99
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



# Parsed testcases at query #100
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



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 5/19 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = 'test_field'
    var_2 = set()
    var_3 = []
    var_4 = []
    var_5 = 'test_field'
    var_6 = 42



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_precord_evolver_persistent_predicate_is_dirty_true. Retrieved 6/13 statements.
# Partially parsed test_precord_evolver_persistent_predicate_not_isinstance_true. Retrieved 7/15 statements.


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
    var_5 = {var_1: var_2}
    var_6 = module_1.pmap(var_5)



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_partial_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.


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
    var_0 = lambda : 42
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
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999
    var_4 = 'z'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_precord_meta_new_creates_class. Retrieved 9/20 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_attr'
    var_2 = 'TestClass'
    var_3 = ()
    var_4 = '_precord_fields'
    var_5 = '_precord_invariants'
    var_6 = '_precord_mandatory_fields'
    var_7 = '_precord_initial_values'
    var_8 = '__slots__'



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'name'
    var_5 = 'age'



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 6/9 statements.


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
    var_1 = 5
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 10
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_invariant_errors. Retrieved 3/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_mandatory_fields. Retrieved 4/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 4/18 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/19 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 2/18 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []

def test_case_0():
    var_0 = 'TestRecord'
    var_1 = {}
    var_2 = 'required_field'
    var_3 = {var_2}
    var_4 = []
    var_5 = []
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestRecord.required_field'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = []
    var_4 = 'error_code_1'
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'error_code_1'

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = set()
    var_3 = []
    var_4 = len(var_0)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'global_error'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_persistent_returns_precord_instance_when_not_dirty. Retrieved 8/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 8/21 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 8/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockPRecord'
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'MockPRecord'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockPRecord.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockPRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = 'error2'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockPRecord'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockPRecord'
    var_4 = 'a'
    var_5 = 1
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 7/14 statements.
# Partially parsed test_precord_evolver_set_with_invalid_field_name. Retrieved 8/14 statements.
# Partially parsed test_precord_evolver_setitem. Retrieved 7/14 statements.
# Partially parsed test_precord_evolver_set_with_factory_fields_filter. Retrieved 13/25 statements.
# Partially parsed test_precord_evolver_set_with_type_error. Retrieved 7/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'name'
    var_7 = 'test_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = 'TestRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'invalid_field'
    var_7 = 'value'
    var_8 = bool(False)
    assert var_8 is True
    var_9 = "'invalid_field' is not among the specified fields for TestRecord"

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'age'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'age'
    var_7 = 25

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
    var_9 = ()
    var_10 = 'TestRecord'
    var_11 = {}
    var_12 = module_0.pmap(var_11)
    var_13 = 'name'
    var_14 = 'test'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'count'
    var_1 = set()
    var_2 = ()
    var_3 = 'TestRecord'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'count'
    var_7 = 'not_a_number_that_converts'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_meta_new_sets_fields. Retrieved 8/14 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 8/14 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 8/14 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 5/10 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 10/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields. Retrieved 8/20 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 42
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 42
    var_6 = ()
    var_7 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 42
    var_6 = ()
    var_7 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestRecord'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = 'base_field'
    var_1 = '_precord_fields'
    var_2 = True
    var_3 = None
    var_4 = 'base_field'
    var_5 = True
    var_6 = None
    var_7 = 'child_field'
    var_8 = False
    var_9 = 10
    var_10 = 'ChildRecord'
    var_11 = 'child_field'
    var_12 = 'base_field'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 2/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 2/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 4/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error1'
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'TestRecord.value'
    var_2 = bool(False)
    assert var_2 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'error1'
    var_2 = 'error2'
    var_3 = 'TestRecord.value'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_slots. Retrieved 9/19 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'test_attr'
    var_3 = ()
    var_4 = 'TestPRecord'
    var_5 = '_precord_fields'
    var_6 = '_precord_invariants'
    var_7 = '_precord_mandatory_fields'
    var_8 = '_precord_initial_values'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_persistent_returns_precord_instance. Retrieved 5/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/15 statements.
# Partially parsed test_persistent_detects_missing_mandatory_fields. Retrieved 6/14 statements.
# Partially parsed test_persistent_calls_global_invariants. Retrieved 4/15 statements.
# Partially parsed test_persistent_with_clean_state_returns_original. Retrieved 5/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = [None] * 8
    var_4 = 0
    var_5 = []

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = [None] * 8
    var_4 = 0
    var_5 = []
    var_6 = 'error1'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = [None] * 8
    var_4 = 0
    var_5 = []
    var_6 = 'field1'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = 'required_field'
    var_3 = {var_2}
    var_4 = []
    var_5 = [None] * 8
    var_6 = 0
    var_7 = []
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'TestPRecord.required_field'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = [None] * 8
    var_3 = 0
    var_4 = []
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = [None] * 8
    var_4 = 0
    var_5 = []



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 5/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 7/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 5/17 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 5/17 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 5/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = module_0.pmap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = module_0.pmap()
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockClass'
    var_6 = module_0.pmap()
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'MockClass.field1'
    var_9 = 'MockClass.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = 'MockClass'
    var_5 = module_0.pmap()
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = module_0.pmap()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 2/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 2/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 4/15 statements.


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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_persistent_evaluates_mandatory_fields_predicate. Retrieved 5/12 statements.
# Partially parsed test_persistent_mandatory_fields_predicate_false_with_empty_set. Retrieved 2/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'mandatory_field'
    var_3 = 'value1'
    var_4 = 'value2'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_returns_precord_instance. Retrieved 4/12 statements.
# Partially parsed test_persistent_with_modified_fields. Retrieved 6/14 statements.
# Partially parsed test_persistent_raises_on_missing_mandatory_fields. Retrieved 5/13 statements.
# Partially parsed test_persistent_raises_on_field_invariant_violation. Retrieved 3/13 statements.
# Partially parsed test_persistent_with_no_dirty_changes. Retrieved 2/9 statements.
# Partially parsed test_persistent_calls_global_invariants. Retrieved 8/21 statements.
# Partially parsed test_persistent_with_factory_field. Retrieved 3/10 statements.
# Partially parsed test_persistent_preserves_unmodified_fields. Retrieved 8/14 statements.


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
    var_4 = 'x'
    var_5 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0.field(mandatory=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 'x'
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'TestRecord.x'

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = -5
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'must_be_positive'

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
    var_4 = 'x'
    var_5 = -5
    var_6 = 'y'
    var_7 = 3
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'sum_must_be_positive'

def test_case_0():
    var_0 = 1
    var_1 = 'x'
    var_2 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 'y'
    var_7 = 20



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_kwargs_override_initial. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 5/12 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.


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
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = None
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = True
    var_3 = 10
    var_4 = 'extra_field'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_precord_meta_new_sets_fields. Retrieved 8/18 statements.
# Partially parsed test_precord_meta_new_mandatory_fields. Retrieved 7/17 statements.
# Partially parsed test_precord_meta_new_initial_values. Retrieved 7/17 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 5/14 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 9/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields. Retrieved 9/22 statements.
# Partially parsed test_precord_meta_new_removes_field_from_dict. Retrieved 7/16 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = ()
    var_6 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 'default_value'
    var_5 = ()
    var_6 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestRecord'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = True
    var_2 = None
    var_3 = 'Parent'
    var_4 = ()
    var_5 = 'child_field'
    var_6 = False
    var_7 = 'child_default'
    var_8 = 'Child'
    var_9 = 'parent_field'
    var_10 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'other_attr'
    var_2 = True
    var_3 = None
    var_4 = 'value'
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = 'other_attr'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 9/26 statements.
# Partially parsed test_persistent_raises_on_missing_mandatory_fields. Retrieved 9/27 statements.
# Partially parsed test_persistent_raises_on_field_invariant_error. Retrieved 10/29 statements.
# Partially parsed test_persistent_raises_on_accumulated_invariant_errors. Retrieved 10/29 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = 0
    var_5 = 'MockCls'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_1}
    var_3 = []
    var_4 = None
    var_5 = 0
    var_6 = 'MockCls'
    var_7 = 'field1'
    var_8 = 'value1'
    var_9 = {var_7: var_8}
    var_10 = module_0.pmap(var_9)
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'MockCls.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = 0
    var_5 = 'MockCls'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = 'error_code_1'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = 0
    var_5 = 'MockCls'
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = 'accumulated_error'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'accumulated_error'



# Parsed testcases at query #15
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
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 'c'
    var_6 = 'MultiFieldRecord'
    var_7 = 'first='
    var_8 = 'second='
    var_9 = 'third='

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = "hello'world"
    var_2 = 'SpecialRecord'
    var_3 = 'text='



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/7 statements.
# Partially parsed test_precord_new_kwargs_override_initial_values. Retrieved 4/7 statements.


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
    var_2 = 'x'
    var_3 = 5
    var_4 = {var_2: var_3}
    var_5 = 15

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = lambda : 100
    var_4 = {var_2: var_3}
    var_5 = 200

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 5
    var_4 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 1
    var_5 = 2
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 100



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_precord_meta_new_sets_precord_fields. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 7/13 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 8/14 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 5/10 statements.
# Partially parsed test_precord_meta_new_sets_invariants. Retrieved 10/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 9/18 statements.
# Partially parsed test_precord_meta_new_removes_pfield_from_dct. Retrieved 5/13 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = ()
    var_6 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default_value'
    var_6 = ()
    var_7 = 'TestRecord'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestRecord'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = 'base_field'
    var_1 = True
    var_2 = None
    var_3 = 'BaseRecord'
    var_4 = ()
    var_5 = 'child_field'
    var_6 = False
    var_7 = 'child'
    var_8 = 'ChildRecord'
    var_9 = 'base_field'
    var_10 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestRecord'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 3/14 statements.
# Partially parsed test_persistent_with_accumulated_invariant_errors. Retrieved 6/16 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = []

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = []
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockCls'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockCls'
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = []
    var_5 = 'field_error'
    var_6 = 'MockCls.required_field'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_correct_attributes. Retrieved 8/21 statements.


def test_case_0():
    var_0 = True
    var_1 = 'test_field'
    var_2 = ()
    var_3 = 'TestPRecord'
    var_4 = '_precord_fields'
    var_5 = 'test_field'
    var_6 = '_precord_mandatory_fields'
    var_7 = 'test_field'
    var_8 = '_precord_initial_values'
    var_9 = '_precord_invariants'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_internal_creation. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_overrides_initial. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_mixed_initial_and_kwargs. Retrieved 5/8 statements.


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
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

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
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20
    var_3 = module_0.field(initial=var_2)
    var_4 = module_0.field()
    var_5 = 30
    var_6 = 40



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 4/13 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_with_different_serializers. Retrieved 4/14 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 3/10 statements.


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
    var_1 = 'alice'
    var_2 = 42
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = None
    var_2 = 'test'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_present. Retrieved 6/13 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'John'
    var_5 = 30



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_persistent_with_mandatory_fields. Retrieved 2/11 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'mandatory_field'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_precord_initial_values_predicate. Retrieved 8/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'value'
    var_4 = 'default_name'
    var_5 = 42
    var_6 = lambda : var_5
    var_7 = 'custom_name'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 6/11 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_initial_values_override. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 1/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_multiple_fields. Retrieved 6/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 1
    var_2 = {var_0: var_1}
    var_3 = module_0.pmap(var_2)
    var_4 = var_3._size
    var_5 = var_3._buckets

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
    var_1 = False
    var_2 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 100
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
    var_0 = 'x'
    var_1 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 8/18 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 7/17 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 6/18 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 3/12 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 7/4 statements.
# Partially parsed test_precord_meta_new_removes_fields_from_dct. Retrieved 5/15 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 7/20 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_mandatory_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = False
    var_3 = 'default_val'
    var_4 = ()
    var_5 = 'TestRecord'
    var_6 = '_precord_initial_values'
    var_7 = 'field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = ()
    var_2 = 'TestRecord'
    var_3 = '__slots__'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = ()
    var_4 = 'TestRecord'
    var_5 = 'field1'
    var_6 = 'field2'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = True
    var_2 = 'ParentRecord'
    var_3 = ()
    var_4 = 'child_field'
    var_5 = False
    var_6 = 'ChildRecord'
    var_7 = 'parent_field'
    var_8 = 'child_field'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 8/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = 4
    var_6 = 5
    var_7 = 6



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_without_serializer. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/10 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 2/9 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_with_mixed_serializers. Retrieved 5/12 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 1/8 statements.


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

def test_case_0():
    var_0 = 'test'
    var_1 = 'json'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 'test'
    var_4 = True

def test_case_0():
    var_0 = None



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_multiple_kwargs. Retrieved 6/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 0
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Alice'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Bob'
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Charlie'
    var_2 = 'value'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'David'
    var_2 = 'value'
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'Eve'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'default_name'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 'Frank'
    var_5 = 25

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



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 6/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 6/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields. Retrieved 7/16 statements.


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



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 4/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 4/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present. Retrieved 7/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = []
    var_4 = 'error1'
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = []
    var_4 = 'TestRecord.x'
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = []
    var_4 = 'error1'
    var_5 = 'error2'
    var_6 = 'TestRecord.x'
    var_7 = 'TestRecord.y'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_special_characters. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_nested_structure. Retrieved 5/9 statements.


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
    var_1 = 42

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello"world'
    var_2 = 'SpecialRecord'
    var_3 = 'text='
    var_4 = 'hello"world'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 'NestedRecord'
    var_6 = 'data='



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 3/12 statements.


def test_case_0():
    var_0 = 'test'
    var_1 = 'name'
    var_2 = 'new_value'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_precord_meta_new_sets_precord_fields. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 6/12 statements.
# Partially parsed test_precord_meta_new_sets_precord_invariants. Retrieved 10/4 statements.
# Partially parsed test_precord_meta_new_removes_pfield_from_dct. Retrieved 5/13 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 9/18 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 42
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 42
    var_6 = ()
    var_7 = 'TestRecord'
    var_8 = '_precord_mandatory_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = False
    var_4 = 42
    var_5 = ()
    var_6 = 'TestRecord'
    var_7 = '_precord_initial_values'
    var_8 = 'field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestRecord'
    var_5 = '__slots__'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestRecord'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestRecord'

def test_case_0():
    var_0 = 'base_field'
    var_1 = True
    var_2 = None
    var_3 = 'BaseRecord'
    var_4 = ()
    var_5 = 'child_field'
    var_6 = False
    var_7 = 10
    var_8 = 'ChildRecord'
    var_9 = 'base_field'
    var_10 = 'child_field'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 7/11 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Failed to parse test_precord_new_with_callable_initial_values.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_overrides_initial_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 2
    var_3 = module_1.pmap(pre_size=var_2)
    var_4 = var_3._buckets
    var_5 = module_1.pmap(pre_size=var_2)
    var_6 = var_5._size

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
    var_2 = set()
    var_3 = 'Jane'
    var_4 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'Bob'
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Default'
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Default'
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 'Override'
    var_4 = 35



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_precord_evolver_set_with_existing_field. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'name'
    var_5 = 'Jane'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_serialize_without_format. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_format. Retrieved 4/13 statements.
# Failed to parse test_serialize_empty_record.
# Partially parsed test_serialize_multiple_fields_with_serializers. Retrieved 5/19 statements.
# Partially parsed test_serialize_with_none_values. Retrieved 4/8 statements.


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
    var_3 = 'upper'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello'
    var_2 = 255
    var_3 = 'unchanged'
    var_4 = 'upper'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = None
    var_3 = 'test'



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 8/19 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = set()
    var_3 = ()
    var_4 = []
    var_5 = 'name'
    var_6 = 'test'
    var_7 = 'value'
    var_8 = 42



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_persistent_predicate_is_dirty_true. Retrieved 4/17 statements.
# Partially parsed test_persistent_predicate_not_isinstance_true. Retrieved 6/16 statements.
# Partially parsed test_persistent_predicate_both_conditions_true. Retrieved 4/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'x'
    var_3 = 1
    var_4 = 0

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = 0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'x'
    var_3 = 42
    var_4 = 0



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 8/22 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 8/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/20 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 8/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = 'error2'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'TestClass'
    var_6 = {}
    var_7 = module_0.pmap(var_6)
    var_8 = bool(False)
    assert var_8 is True
    var_9 = 'TestClass.field1'
    var_10 = 'TestClass.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'MockCls'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_precord_new_predicate_false. Retrieved 9/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'Jane'
    var_5 = 25
    var_6 = 10
    var_7 = 'Bob'
    var_8 = 35



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 6/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = 'test_field'
    var_2 = set()
    var_3 = []
    var_4 = module_0.pmap()
    var_5 = 'test_field'
    var_6 = 42



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
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
    var_2 = True
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Unknown'
    var_1 = module_0.field(initial=var_0)

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



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/14 statements.


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



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_multiple_fields_order. Retrieved 6/10 statements.
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
    var_4 = 2
    var_5 = 3
    var_6 = 'MultiFieldRecord'
    var_7 = 'x=1'
    var_8 = 'y=2'
    var_9 = 'z=3'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'hello"world'
    var_2 = 'SpecialRecord'
    var_3 = 'text='
    var_4 = 'hello'



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 5/9 statements.
# Partially parsed test_precord_new_with_initial_values_and_override. Retrieved 5/9 statements.
# Partially parsed test_precord_new_empty. Retrieved 2/6 statements.
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
    var_1 = module_0.field()
    var_2 = 5
    var_3 = 15
    var_4 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 100
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 42
    var_4 = lambda : var_3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field()
    var_2 = 'x'
    var_3 = 100
    var_4 = 200

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
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_precord_meta_new_sets_precord_fields. Retrieved 5/12 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 7/13 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 8/15 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 5/10 statements.
# Partially parsed test_precord_meta_new_sets_precord_invariants. Retrieved 9/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_base. Retrieved 9/18 statements.
# Partially parsed test_precord_meta_new_removes_field_from_dct. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestClass'
    var_5 = '_precord_fields'
    var_6 = 'field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = ()
    var_6 = 'TestClass'
    var_7 = '_precord_mandatory_fields'
    var_8 = 'field1'
    var_9 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = 'default1'
    var_4 = False
    var_5 = None
    var_6 = ()
    var_7 = 'TestClass'
    var_8 = '_precord_initial_values'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestClass'
    var_5 = '__slots__'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestClass'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = True
    var_6 = None
    var_7 = ()
    var_8 = 'TestClass'
    var_9 = '_precord_invariants'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 'base_value'
    var_3 = 'BaseClass'
    var_4 = ()
    var_5 = 'field2'
    var_6 = False
    var_7 = None
    var_8 = 'DerivedClass'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestClass'
    var_5 = False



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_ignore_extra. Retrieved 4/7 statements.
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
    var_1 = 'x'

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
    var_1 = 5
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = True
    var_4 = 'z'

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



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_persistent_predicate_line_1_is_dirty_true. Retrieved 3/11 statements.
# Partially parsed test_persistent_predicate_line_1_isinstance_false. Retrieved 3/16 statements.
# Partially parsed test_persistent_predicate_line_6_evaluates_true_when_dirty. Retrieved 4/17 statements.


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
    var_2 = '__bases__'
    var_3 = 0
    var_4 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = 'x'
    var_3 = 100
    var_4 = 0



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 6/16 statements.


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



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_precord_repr. Retrieved 5/11 statements.


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



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_values. Retrieved 4/9 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30
    var_4 = 'name'
    var_5 = 'age'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
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
    var_0 = module_0.field()
    var_1 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'

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
    var_2 = 1
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/11 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/13 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 6/15 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 3/14 statements.
# Partially parsed test_persistent_successful_with_no_invariants. Retrieved 4/10 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = []

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'TestClass'
    var_6 = []
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = []
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = bool(False)
    assert var_7 is True

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'TestClass'
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = []



# Parsed testcases at query #57
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_kwargs_override_initial. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.


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
    var_1 = module_0.field()

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
    var_3 = {var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True
    var_4 = 'y'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/8 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_new_kwargs_override_initial_values. Retrieved 3/6 statements.


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
    var_3 = 15

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 50
    var_3 = 999
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = 100



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_override_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_callable_initial. Retrieved 1/4 statements.
# Failed to parse test_precord_new_empty.
# Partially parsed test_precord_new_invalid_field. Retrieved 2/6 statements.


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
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 7
    var_2 = False

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 10
    var_1 = module_0.field(initial=var_0)
    var_2 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : 99
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 5
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'y'



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.


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
    var_1 = 'x'
    var_2 = '42'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 0
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Bob'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Charlie'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'David'
    var_3 = 25
    var_4 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Eve'
    var_2 = 'ignored'
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'default'
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'Frank'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'default_name'
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 'Grace'
    var_4 = 28



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_partial_kwargs. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Unknown'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Unknown'
    var_1 = module_0.field(initial=var_0)
    var_2 = 0
    var_3 = module_0.field(initial=var_2)
    var_4 = 'Bob'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = True
    var_3 = 'Charlie'
    var_4 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 'David'
    var_3 = 'ignored'
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'Default'
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'Eve'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'test'
    var_3 = 42



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 8/18 statements.


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
    var_8 = [var_7]



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_multiple_kwargs. Retrieved 6/9 statements.


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
    var_1 = 'x'

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = None

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
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/9 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.


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

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = var_1[0]
    assert var_2 == 1

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
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = {}



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 2/20 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = []
    var_2 = False



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 25
    var_2 = module_0.field(initial=var_1)
    var_3 = 'Jane'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Test'
    var_2 = 'name'
    var_3 = [var_2]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Test'
    var_2 = 'value'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'Test'
    var_2 = 'value'
    var_3 = True
    var_4 = 'extra_field'

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'Direct'
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._size
    var_6 = var_4._buckets



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Failed to parse test_precord_constructor_empty_record.
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
    var_0 = lambda : 42
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 5
    var_3 = {var_1: var_2}
    var_4 = module_1.pmap(var_3)
    var_5 = var_4._precord_size
    var_6 = var_4._precord_buckets

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = True
    var_4 = 'z'

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



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_persistent_predicate_line_6_true_when_is_dirty. Retrieved 5/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = set()
    var_2 = ()
    var_3 = []
    var_4 = 'x'
    var_5 = 42



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_persistent_predicate_line_6_evaluates_to_false. Retrieved 9/17 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = 'Alice'
    var_5 = 30
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_1.pmap(var_6)



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/15 statements.


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



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_internal_attributes. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/9 statements.
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
    var_2 = module_0.field()
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = bool(False)
    assert var_3 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = True
    var_2 = 2
    var_3 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'x'
    var_2 = 2
    var_3 = lambda v: v * var_2
    var_4 = {var_1: var_3}
    var_5 = 5

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



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 3/6 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = lambda : 20
    var_2 = module_0.field(initial=var_1)
    var_3 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 3
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = {}

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

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 5
    var_1 = module_0.field(initial=var_0)
    var_2 = 10
    var_3 = module_0.field(initial=var_2)
    var_4 = 100



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/15 statements.


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



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 10/22 statements.
# Partially parsed test_precord_meta_new_sets_mandatory_fields. Retrieved 10/22 statements.
# Partially parsed test_precord_meta_new_sets_initial_values. Retrieved 10/23 statements.
# Partially parsed test_precord_meta_new_sets_slots. Retrieved 7/18 statements.
# Partially parsed test_precord_meta_new_stores_invariants. Retrieved 11/4 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 10/24 statements.
# Partially parsed test_precord_meta_new_removes_field_from_dct. Retrieved 6/19 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '__module__'
    var_3 = True
    var_4 = None
    var_5 = False
    var_6 = 42
    var_7 = 'test_module'
    var_8 = 'TestRecord'
    var_9 = '_precord_fields'
    var_10 = 'field1'
    var_11 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '__module__'
    var_3 = True
    var_4 = None
    var_5 = False
    var_6 = 42
    var_7 = 'test_module'
    var_8 = 'TestRecord'
    var_9 = '_precord_mandatory_fields'
    var_10 = 'field1'
    var_11 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '__module__'
    var_3 = True
    var_4 = None
    var_5 = False
    var_6 = 42
    var_7 = 'test_module'
    var_8 = 'TestRecord'
    var_9 = '_precord_initial_values'
    var_10 = 'field1'

def test_case_0():
    var_0 = 'field1'
    var_1 = '__module__'
    var_2 = True
    var_3 = None
    var_4 = 'test_module'
    var_5 = 'TestRecord'
    var_6 = '__slots__'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = '__module__'
    var_6 = True
    var_7 = None
    var_8 = 'test_module'
    var_9 = 'TestRecord'
    var_10 = '_precord_invariants'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field1'
    var_4 = '__invariant__'
    var_5 = '__module__'
    var_6 = True
    var_7 = None
    var_8 = 'test_module'
    var_9 = 'TestRecord'
    var_10 = '_precord_invariants'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = '__module__'
    var_2 = True
    var_3 = None
    var_4 = 'test_module'
    var_5 = 'ParentRecord'
    var_6 = 'child_field'
    var_7 = False
    var_8 = 10
    var_9 = 'ChildRecord'
    var_10 = 'parent_field'
    var_11 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = '__module__'
    var_2 = True
    var_3 = None
    var_4 = 'test_module'
    var_5 = 'TestRecord'



# Parsed testcases at query #78
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



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 8/20 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_error_codes. Retrieved 8/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/18 statements.
# Partially parsed test_persistent_successful_with_all_valid. Retrieved 8/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = 'error2'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockCls'
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
    var_2 = 'MockCls'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #80
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #81
#--------------------------

# Partially parsed test_serialize_method_exists_and_callable. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 'serialize'



# Parsed testcases at query #82
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 4/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 4/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields_present. Retrieved 6/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = frozenset()
    var_2 = ()
    var_3 = []
    var_4 = 'error_code_1'
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = frozenset()
    var_2 = ()
    var_3 = []
    var_4 = 'TestRecord.name'
    var_5 = bool(False)
    assert var_5 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = frozenset()
    var_2 = ()
    var_3 = []
    var_4 = 'error_code_1'
    var_5 = 'error_code_2'
    var_6 = 'TestRecord.name'
    var_7 = bool(False)
    assert var_7 is True



# Parsed testcases at query #83
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 6/19 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 6/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_error_codes_and_missing_fields_present. Retrieved 8/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = module_0.pmap()
    var_5 = 'error1'
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = module_0.pmap()
    var_5 = 'MockClass.field1'
    var_6 = bool(False)
    assert var_6 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = module_0.pmap()
    var_5 = 'error1'
    var_6 = 'error2'
    var_7 = 'MockClass.field1'
    var_8 = bool(False)
    assert var_8 is True



# Parsed testcases at query #84
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 10/15 statements.


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



# Parsed testcases at query #85
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 6/22 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_error_codes. Retrieved 8/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 6/18 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 5/20 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = {}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockCls'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error1'
    var_7 = 'error2'
    var_8 = bool(False)
    assert var_8 is True

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'MockCls'
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
    var_2 = 'MockCls'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'



# Parsed testcases at query #86
#--------------------------

# Partially parsed test_precord_repr. Retrieved 4/8 statements.
# Failed to parse test_precord_repr_empty.
# Partially parsed test_precord_repr_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_multiple_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_repr_with_nested_structure. Retrieved 4/8 statements.


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
    var_7 = 'a=1'
    var_8 = "b='two'"
    var_9 = 'c=3.0'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'nested'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = 'NestedRecord'
    var_5 = 'data='
    var_6 = 'nested'



# Parsed testcases at query #87
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/13 statements.
# Partially parsed test_precord_new_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_new_empty. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values_override. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_invalid_field_raises_error. Retrieved 2/6 statements.


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
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = 10
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.field()
    var_6 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 5
    var_3 = 10
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
    var_1 = 1
    var_2 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = bool(False)
    assert var_2 is True



# Parsed testcases at query #88
#--------------------------

# Partially parsed test_precord_meta_new_creates_precord_fields. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_creates_mandatory_fields_set. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_creates_initial_values_dict. Retrieved 9/16 statements.
# Partially parsed test_precord_meta_new_sets_empty_slots. Retrieved 6/12 statements.
# Partially parsed test_precord_meta_new_creates_invariants. Retrieved 7/19 statements.
# Partially parsed test_precord_meta_new_inherits_fields_from_bases. Retrieved 10/22 statements.
# Partially parsed test_precord_meta_new_removes_field_definitions_from_dict. Retrieved 8/22 statements.


def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestPRecord'
    var_8 = '_precord_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestPRecord'
    var_8 = '_precord_mandatory_fields'
    var_9 = 'field1'
    var_10 = 'field2'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestPRecord'
    var_8 = '_precord_initial_values'

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = None
    var_3 = ()
    var_4 = 'TestPRecord'
    var_5 = '__slots__'

def test_case_0():
    var_0 = 'field1'
    var_1 = '__invariant__'
    var_2 = True
    var_3 = None
    var_4 = ()
    var_5 = 'TestPRecord'
    var_6 = '_precord_invariants'

def test_case_0():
    var_0 = 'parent_field'
    var_1 = '_precord_fields'
    var_2 = True
    var_3 = None
    var_4 = 'Parent'
    var_5 = ()
    var_6 = 'child_field'
    var_7 = False
    var_8 = 'default'
    var_9 = 'Child'
    var_10 = 'child_field'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = None
    var_4 = False
    var_5 = 'default'
    var_6 = ()
    var_7 = 'TestPRecord'



# Parsed testcases at query #89
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_present. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_missing_fields_present. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_both_errors_and_missing_fields. Retrieved 6/14 statements.


import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = module_1.pmap(var_1)
    var_3 = 'error1'
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = module_1.pmap(var_1)
    var_3 = 'missing_field'
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0
import pyrsistent._pmap as module_1

def test_case_0():
    var_0 = module_0.field()
    var_1 = {}
    var_2 = module_1.pmap(var_1)
    var_3 = 'error1'
    var_4 = 'error2'
    var_5 = 'missing1'
    var_6 = bool(False)
    assert var_6 is True



# Parsed testcases at query #90
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_without_ignore_extra. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_from_existing_instance. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/8 statements.


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
    var_1 = 'name'
    var_2 = 'extra'
    var_3 = 'John'
    var_4 = 'value'
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = True
    var_7 = 'extra'

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
    var_1 = module_0.field()
    var_2 = 'John'
    var_3 = 30

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'name'
    var_2 = 'John'
    var_3 = {var_1: var_2}
    var_4 = None



# Parsed testcases at query #91
#--------------------------

# Partially parsed test_precord_evolver_set_with_existing_field. Retrieved 4/14 statements.


def test_case_0():
    var_0 = 'John'
    var_1 = 30
    var_2 = 'name'
    var_3 = 'Jane'



# Parsed testcases at query #92
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_internal_params. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
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
    var_1 = 'x'

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
    var_3 = True
    var_4 = 'y'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 'x'



# Parsed testcases at query #93
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
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
    var_4 = 'Jane'
    var_5 = 25

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'John'
    var_2 = 'value'
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

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
    var_1 = bool(False)
    assert var_1 is True

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



# Parsed testcases at query #94
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
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
    var_0 = lambda : [1, 2, 3]
    var_1 = module_0.field(initial=var_0)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = True
    var_3 = 999
    var_4 = 'extra_field'

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



# Parsed testcases at query #95
#--------------------------

# Partially parsed test_persistent_with_no_changes. Retrieved 6/21 statements.
# Partially parsed test_persistent_with_mandatory_fields_missing. Retrieved 6/18 statements.
# Partially parsed test_persistent_with_field_invariant_error. Retrieved 7/20 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 5/20 statements.
# Partially parsed test_persistent_dirty_state_creates_new_instance. Retrieved 8/21 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'required_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = 'TestClass'
    var_5 = {}
    var_6 = module_0.pmap(var_5)
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'TestClass.required_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = module_0.pmap(var_4)
    var_6 = 'error_code_1'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = 'TestClass'
    var_3 = {}
    var_4 = module_0.pmap(var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'global_error'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'TestClass'
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = module_0.pmap(var_6)



# Parsed testcases at query #96
#--------------------------

# Partially parsed test_persistent_checks_mandatory_fields_when_precord_mandatory_fields_is_truthy. Retrieved 4/15 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'name'
    var_3 = 'age'
    var_4 = bool(False)
    assert var_4 is True



# Parsed testcases at query #97
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/15 statements.


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



# Parsed testcases at query #98
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



# Parsed testcases at query #99
#--------------------------

# Partially parsed test_precord_repr. Retrieved 6/12 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = 'TestRecord('
    var_5 = ')'
    var_6 = 'name='
    var_7 = 'age='
    var_8 = "'Alice'"
    var_9 = '30'



# Parsed testcases at query #100
#--------------------------

# Partially parsed test_persistent_returns_result_when_no_errors. Retrieved 8/25 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 9/27 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_error. Retrieved 10/29 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 8/29 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = None
    var_4 = 0
    var_5 = 'field1'
    var_6 = 'value1'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field2'
    var_4 = {var_3}
    var_5 = []
    var_6 = None
    var_7 = 0
    var_8 = 'field1'
    var_9 = 'value1'
    var_10 = {var_8: var_9}
    var_11 = module_0.pmap(var_10)
    var_12 = bool(False)
    assert var_12 is True
    var_13 = 'MockCls.field2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = 'field1'
    var_2 = set()
    var_3 = []
    var_4 = None
    var_5 = 0
    var_6 = 'field1'
    var_7 = 'value1'
    var_8 = {var_6: var_7}
    var_9 = module_0.pmap(var_8)
    var_10 = 'error_code_1'
    var_11 = bool(False)
    assert var_11 is True
    var_12 = 'error_code_1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = 'field1'
    var_2 = set()
    var_3 = None
    var_4 = 0
    var_5 = 'field1'
    var_6 = 'value1'
    var_7 = {var_5: var_6}
    var_8 = module_0.pmap(var_7)
    var_9 = bool(False)
    assert var_9 is True
    var_10 = 'global_error'



# Parsed testcases at query #101
#--------------------------

# Partially parsed test_precord_evolver_set_with_valid_field. Retrieved 2/15 statements.
# Partially parsed test_precord_evolver_set_with_invalid_field_name. Retrieved 3/16 statements.
# Partially parsed test_precord_evolver_set_with_factory_fields_filter. Retrieved 2/17 statements.
# Partially parsed test_precord_evolver_set_with_field_not_in_factory_fields. Retrieved 2/17 statements.
# Partially parsed test_precord_evolver_setitem_calls_set. Retrieved 2/14 statements.


def test_case_0():
    var_0 = 'test_field'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42

def test_case_0():
    var_0 = 'MockClass'
    var_1 = 'valid_field'
    var_2 = []
    var_3 = 'invalid_field'
    var_4 = 42
    var_5 = bool(False)
    assert var_5 is True
    var_6 = 'invalid_field'
    var_7 = 'MockClass'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = []
    var_3 = 'field1'
    var_4 = 42

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = []
    var_3 = 'field2'
    var_4 = 42

def test_case_0():
    var_0 = 'test_field'
    var_1 = []
    var_2 = 'test_field'
    var_3 = 42



# Parsed testcases at query #102
#--------------------------

# Partially parsed test_persistent_returns_instance_when_not_dirty. Retrieved 4/21 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_missing_fields. Retrieved 4/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_with_field_invariant_errors. Retrieved 6/20 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 3/19 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 4/17 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []

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
    var_8 = 'MockClass.field1'
    var_9 = 'MockClass.field2'

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
    var_1 = set()
    var_2 = 'MockClass'
    var_3 = []
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'global_error'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'MockClass'
    var_4 = []



# Parsed testcases at query #103
#--------------------------

# Partially parsed test_precord_meta_new_basic. Retrieved 4/12 statements.
# Partially parsed test_precord_meta_new_with_fields. Retrieved 4/11 statements.
# Partially parsed test_precord_meta_new_with_invariant. Retrieved 5/4 statements.
# Partially parsed test_precord_meta_new_inheritance. Retrieved 1/8 statements.
# Partially parsed test_precord_meta_new_multiple_mandatory_fields. Retrieved 3/12 statements.
# Partially parsed test_precord_meta_new_empty_initial_values. Retrieved 2/10 statements.


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'

def test_case_0():
    var_0 = True
    var_1 = 42
    var_2 = False
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'field2'
    var_8 = 'field1'

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = 'field1'

def test_case_0():
    pass

def test_case_0():
    var_0 = True
    var_1 = 10
    var_2 = False
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = 'field3'

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'field1'
    var_3 = 'field2'



# Parsed testcases at query #104
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2



# Parsed testcases at query #105
#--------------------------

# Partially parsed test_precord_meta_new_creates_class_with_slots. Retrieved 5/11 statements.


def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = 'test_attr'
    var_3 = ()
    var_4 = 'TestPRecord'



# Parsed testcases at query #106
#--------------------------

# Partially parsed test_precord_constructor_with_kwargs. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_override_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_true. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_partial_fields. Retrieved 5/8 statements.


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
    var_4 = 999
    var_5 = 888

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
    var_2 = 999
    var_3 = False
    var_4 = bool(False)
    assert var_4 is True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 999
    var_3 = True
    var_4 = 'z'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 3
    var_5 = 'y'



