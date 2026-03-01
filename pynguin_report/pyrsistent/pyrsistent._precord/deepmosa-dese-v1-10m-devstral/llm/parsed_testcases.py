####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 5/10 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 6/10 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 6/12 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 5/10 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 3/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = var_0._buckets
    var_2 = module_0.PMap()
    var_3 = var_2._size
    var_4 = None
    var_5 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = module_0.PMap()
    var_3 = None
    var_4 = False
    var_5 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'error1'
    var_4 = 'error2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_new_with_empty_bases_and_dct. Retrieved 4/5 statements.
# Partially parsed test_new_with_fields_in_dct. Retrieved 12/13 statements.
# Partially parsed test_new_with_inherited_fields. Retrieved 6/10 statements.
# Partially parsed test_new_with_invariant_in_dct. Retrieved 7/4 statements.
# Partially parsed test_new_with_inherited_invariant. Retrieved 6/4 statements.
# Partially parsed test_new_with_multiple_invariants. Retrieved 5/4 statements.
# Partially parsed test_new_with_non_callable_invariant_raises_type_error. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = module_0._PField(var_2)
    var_4 = 42
    var_5 = module_0._PField(var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'TestClass'
    var_8 = ()
    var_9 = module_0._PField(var_2)
    var_10 = module_0._PField(var_4)
    var_11 = {var_0: var_9, var_1: var_10}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}
    var_2 = 'field1'
    var_3 = True
    var_4 = module_0._PField(var_3)
    var_5 = {var_2: var_4}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = 'TestClass'
    var_4 = {}
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = 'Test1'
    var_2 = (var_0, var_1)

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = 'TestClass'
    var_4 = {}

def test_case_0():
    var_0 = True
    var_1 = 'Test2'
    var_2 = (var_0, var_1)
    var_3 = 'TestClass'
    var_4 = {}

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '__invariant__'
    var_3 = 'not callable'
    var_4 = {var_2: var_3}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_persistent_returns_same_instance_when_not_dirty_and_correct_type. Retrieved 4/7 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/11 statements.
# Partially parsed test_persistent_creates_new_instance_when_not_correct_type. Retrieved 7/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_missing_mandatory_fields. Retrieved 4/8 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_field_invariants. Retrieved 5/10 statements.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 4/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = []
    var_5 = module_0.PMap()
    var_6 = 'field'
    var_7 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'missing_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = None
    var_2 = {var_0: var_1}
    var_3 = set()
    var_4 = []
    var_5 = module_0.PMap()
    var_6 = 'error_code'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = module_0.PMap()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 9/11 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 5/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 9/11 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 6/9 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 7/10 statements.


def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = None
    var_4 = 1
    var_5 = (var_4, var_2)
    var_6 = [var_5]
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_2, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '_factory_fields'
    var_3 = 1
    var_4 = 2
    var_5 = [var_0, var_1]
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = '_ignore_extra'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = True
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = {var_1: var_4}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : var_2
    var_4 = 2
    var_5 = 3
    var_6 = {var_1: var_5}



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_precord_mandatory_fields_contains_only_mandatory_fields.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_precord_initial_values_present. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : [1, 2, 3]
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 5
    var_6 = {var_4: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_repr_empty.
# Partially parsed test_repr_single_field. Retrieved 1/5 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 2/7 statements.
# Partially parsed test_repr_with_complex_values. Retrieved 7/12 statements.


def test_case_0():
    var_0 = 'value1'

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'

def test_case_0():
    var_0 = 'nested'
    var_1 = 'dict'
    var_2 = {var_0: var_1}
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = [var_3, var_4, var_5]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_repr_format. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'TestRecord('
    var_3 = ')'



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_precord_mandatory_fields_predicate.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_returns_same_instance_when_not_dirty_and_correct_type. Retrieved 4/7 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/11 statements.
# Partially parsed test_persistent_creates_new_instance_when_wrong_type. Retrieved 4/9 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_missing_mandatory_fields. Retrieved 4/8 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_field_invariant_failures. Retrieved 5/11 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_global_invariant_failures. Retrieved 4/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'key'
    var_5 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'GLOBAL_ERROR')
    var_3 = [var_2]
    var_4 = module_0.PMap()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 8/20 statements.
# Partially parsed test_set_with_valid_field_no_factory. Retrieved 7/16 statements.
# Partially parsed test_set_with_invalid_type. Retrieved 8/17 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 7/15 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 5/11 statements.
# Partially parsed test_set_with_ignore_extra_true. Retrieved 7/18 statements.
# Partially parsed test_set_with_ignore_extra_false. Retrieved 8/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'processed_value'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = 'field_name'
    var_5 = module_0.PMap()
    var_6 = False
    var_7 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field_name'
    var_4 = module_0.PMap()
    var_5 = False
    var_6 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = (var_0, var_1)
    var_3 = 'field_name'
    var_4 = module_0.PMap()
    var_5 = False
    var_6 = 'field_name'
    var_7 = 'not_an_int'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'INVALID'
    var_2 = (var_0, var_1)
    var_3 = 'field_name'
    var_4 = module_0.PMap()
    var_5 = None
    var_6 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'nonexistent_field'
    var_4 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'processed_value'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = 'field_name'
    var_5 = module_0.PMap()
    var_6 = 42

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'processed_value'
    var_1 = True
    var_2 = None
    var_3 = (var_1, var_2)
    var_4 = 'field_name'
    var_5 = module_0.PMap()
    var_6 = False
    var_7 = 42



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_predicate_at_line_5_evaluates_to_false.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_new_with_fields_and_no_invariants. Retrieved 10/13 statements.
# Partially parsed test_new_with_invariant. Retrieved 6/4 statements.
# Partially parsed test_new_with_inherited_fields. Retrieved 7/10 statements.
# Partially parsed test_new_with_inherited_invariants. Retrieved 2/12 statements.
# Partially parsed test_new_with_multiple_inherited_invariants. Retrieved 1/16 statements.
# Partially parsed test_new_with_invariant_returning_multiple_results. Retrieved 3/11 statements.


def test_case_0():
    var_0 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field3'
    var_4 = module_0._PField()
    var_5 = True
    var_6 = module_0._PField(var_5)
    var_7 = 42
    var_8 = module_0._PField(var_7)
    var_9 = {var_1: var_4, var_2: var_6, var_3: var_8}

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = 0
    var_5 = set()

def test_case_0():
    var_0 = True
    var_1 = 'Test'
    var_2 = (var_0, var_1)
    var_3 = len(var_0)
    assert var_3 == 1
    var_4 = 0
    var_5 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0._PField()
    var_4 = True
    var_5 = module_0._PField(var_4)
    var_6 = {var_1: var_3, var_2: var_5}

def test_case_0():
    var_0 = 0
    var_1 = set()

def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = 'not callable'

def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = set()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 5/11 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 3/8 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_field_invariant_failure. Retrieved 5/10 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 7/13 statements.


def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'value1'

def test_case_0():
    var_0 = 'value1'
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = False

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'invalid_value'

def test_case_0():
    var_0 = {}
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'invalid_value'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_instance. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_clean_and_instance. Retrieved 4/7 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = var_0._buckets
    var_2 = module_0.PMap()
    var_3 = var_2._size

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'invalid_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'value1'
    var_3 = 'field2'
    var_4 = 'value2'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 2/6 statements.
# Partially parsed test_set_with_invalid_field. Retrieved 2/7 statements.
# Partially parsed test_set_with_factory_ignore_extra. Retrieved 3/7 statements.
# Partially parsed test_set_with_invariant_exception. Retrieved 2/6 statements.
# Partially parsed test_set_with_type_error. Retrieved 2/7 statements.


def test_case_0():
    var_0 = 'field'
    var_1 = 'value'

def test_case_0():
    var_0 = 'invalid_field'
    var_1 = 'value'

def test_case_0():
    var_0 = True
    var_1 = 'field'
    var_2 = 'value'

def test_case_0():
    var_0 = 'field'
    var_1 = 'value'

def test_case_0():
    var_0 = 'field'
    var_1 = 123



# Parsed testcases at query #3
#--------------------------

# Partially parsed test__new__sets_precord_fields. Retrieved 2/6 statements.
# Partially parsed test__new__inherits_fields. Retrieved 1/5 statements.
# Failed to parse test__new__sets_precord_invariants.
# Failed to parse test__new__inherits_invariants.
# Partially parsed test__new__sets_precord_mandatory_fields. Retrieved 1/4 statements.
# Partially parsed test__new__sets_precord_initial_values. Retrieved 1/4 statements.
# Failed to parse test__new__sets_empty_slots.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'field1'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_persistent_raises_exception_when_invariant_errors_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'missing1'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 5/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = {}
    var_3 = set()
    var_4 = []



# Parsed testcases at query #7
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 5/13 statements.
# Partially parsed test_set_with_invalid_field_type. Retrieved 5/14 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 4/8 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 5/13 statements.
# Partially parsed test_set_with_ignore_extra_compliant_field. Retrieved 6/14 statements.
# Partially parsed test_set_with_ignore_extra_non_compliant_field. Retrieved 6/14 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x * 2
    var_1 = lambda x: (True, None)
    var_2 = 'field1'
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: (True, None)
    var_2 = 'field1'
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 'not_an_int'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMap()
    var_2 = 'nonexistent_field'
    var_3 = 123

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: (False, 'INVALID')
    var_2 = 'field1'
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x, ignore_extra=False: x
    var_1 = lambda x: (True, None)
    var_2 = 'field1'
    var_3 = module_0.PMap()
    var_4 = True
    var_5 = 'field1'
    var_6 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: (True, None)
    var_2 = 'field1'
    var_3 = module_0.PMap()
    var_4 = True
    var_5 = 'field1'
    var_6 = 5



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 5/9 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 3/6 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 5/9 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 5/9 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 7/12 statements.
# Partially parsed test_persistent_with_valid_data. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'value1'

def test_case_0():
    var_0 = 'value1'
    var_1 = None
    var_2 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'invalid_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = 'field2'
    var_6 = 'value2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = None
    var_2 = False
    var_3 = 'field1'
    var_4 = 'valid_value'
    var_5 = 'mandatory_field'
    var_6 = 'required_value'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_without_format. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format. Retrieved 4/8 statements.
# Partially parsed test_serialize_with_no_serializer. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 'upper'
    var_3 = module_0.serialize(var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.serialize()



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_missing_fields_added_when_mandatory_fields_exist. Retrieved 3/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'TestRecord'
    var_4 = module_0.PMap()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 12/13 statements.
# Partially parsed test_precord_new_without_special_attributes. Retrieved 5/9 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 7/11 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 7/12 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 9/14 statements.


def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = 'b'
    var_8 = (var_7, var_2)
    var_9 = (var_8,)
    var_10 = [var_6, var_9]
    var_11 = {var_0: var_2, var_1: var_10}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2
    var_5 = {var_0}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 10
    var_4 = 20
    var_5 = 1
    var_6 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 10
    var_4 = lambda : var_3
    var_5 = 20
    var_6 = lambda : var_5
    var_7 = 1
    var_8 = 2



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_repr_with_empty_record.
# Partially parsed test_repr_with_single_field. Retrieved 1/5 statements.
# Partially parsed test_repr_with_multiple_fields. Retrieved 3/9 statements.
# Partially parsed test_repr_with_nested_record. Retrieved 2/10 statements.
# Partially parsed test_repr_with_special_characters. Retrieved 1/5 statements.
# Partially parsed test_repr_with_none_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = 'value1'

def test_case_0():
    var_0 = 'value1'
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = 'inner_value'
    var_1 = 'outer_value'

def test_case_0():
    var_0 = "value with spaces and 'quotes'"

def test_case_0():
    var_0 = None



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_constructor_with_initial_values. Retrieved 7/8 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/5 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.PRecord(**var_9)
    var_11 = len(var_10)
    assert var_11 == 2

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default1'
    var_3 = 'default2'
    var_4 = 'override1'
    var_5 = {var_0: var_4}
    var_6 = module_0.PRecord(**var_5)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = module_0.PRecord()

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = module_0.PRecord()
    var_1 = len(var_0)
    assert var_1 == 0



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_new_with_no_bases_and_empty_dct. Retrieved 4/5 statements.
# Partially parsed test_new_with_single_base_and_empty_dct. Retrieved 3/7 statements.
# Partially parsed test_new_with_fields_in_dct. Retrieved 10/12 statements.
# Partially parsed test_new_with_invariant_in_dct. Retrieved 4/12 statements.
# Partially parsed test_new_with_inherited_fields. Retrieved 6/12 statements.
# Partially parsed test_new_with_inherited_invariants. Retrieved 2/14 statements.
# Partially parsed test_new_with_non_callable_invariant. Retrieved 5/7 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = set()

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}
    var_2 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = True
    var_1 = module_0._PField(var_0, var_0)
    var_2 = False
    var_3 = 2
    var_4 = module_0._PField(var_3, var_2)
    var_5 = 'TestClass'
    var_6 = ()
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = {var_7: var_1, var_8: var_4}

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '__invariant__'
    var_3 = 0

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = False
    var_1 = 2
    var_2 = module_0._PField(var_1, var_0)
    var_3 = 'TestClass'
    var_4 = 'field2'
    var_5 = {var_4: var_2}

def test_case_0():
    var_0 = 'TestClass'
    var_1 = '__invariant__'

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '__invariant__'
    var_3 = 'not callable'
    var_4 = {var_2: var_3}



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_repr_empty_record.
# Partially parsed test_repr_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 6/9 statements.
# Partially parsed test_repr_with_quoted_strings. Retrieved 2/5 statements.
# Partially parsed test_repr_with_nested_structure. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'value1'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 'value1'
    var_4 = 2
    var_5 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = "John's record"

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_serialize_returns_dict_with_serialized_fields. Retrieved 3/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 5
    var_2 = module_0.serialize()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_set_fields_called_with_correct_arguments. Retrieved 2/4 statements.


def test_case_0():
    var_0 = {}
    var_1 = '_precord_fields'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'test_key'
    var_5 = 'test_value'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_precord_initial_values_used. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = None
    var_8 = None
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_predicate_false. Retrieved 11/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockCls'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = None



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_repr_format. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = 'TestRecord('
    var_3 = ')'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 10/11 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 4/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 5/6 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 5
    var_3 = 1
    var_4 = 2
    var_5 = 3
    var_6 = 4
    var_7 = [var_3, var_4, var_5, var_6, var_2]
    var_8 = {var_0: var_2, var_1: var_7}
    var_9 = module_0.PRecord(**var_8)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = module_0.PRecord(**var_2)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = lambda : var_1
    var_3 = {var_0: var_2}
    var_4 = module_0.PRecord(**var_3)



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_precord_metaclass_initialization. Retrieved 5/12 statements.


def test_case_0():
    var_0 = '_precord_fields'
    var_1 = '_precord_invariants'
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_initial_values'
    var_4 = '__slots__'



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_persistent_raises_when_invariant_error_codes_or_missing_fields. Retrieved 8/17 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = lambda self, value: (True, None)
    var_3 = {}
    var_4 = set()
    var_5 = []
    var_6 = 'error1'
    var_7 = 'field1'



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_precord_initial_values_are_used. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_new_with_empty_bases_and_dct. Retrieved 4/5 statements.
# Partially parsed test_new_with_fields_in_dct. Retrieved 12/13 statements.
# Partially parsed test_new_with_inherited_fields. Retrieved 6/10 statements.
# Partially parsed test_new_with_invariant. Retrieved 7/4 statements.
# Partially parsed test_new_with_inherited_invariant. Retrieved 4/11 statements.
# Partially parsed test_new_with_non_callable_invariant_raises_type_error. Retrieved 5/7 statements.
# Partially parsed test_new_with_multiple_invariants. Retrieved 3/15 statements.
# Partially parsed test_new_with_inherited_multiple_invariants. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = True
    var_3 = module_0._PField(var_2)
    var_4 = 42
    var_5 = module_0._PField(var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = 'TestClass'
    var_8 = ()
    var_9 = module_0._PField(var_2)
    var_10 = module_0._PField(var_4)
    var_11 = {var_0: var_9, var_1: var_10}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}
    var_2 = 'field1'
    var_3 = True
    var_4 = module_0._PField(var_3)
    var_5 = {var_2: var_4}

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = 0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = '__invariant__'
    var_4 = 'TestClass'
    var_5 = ()
    var_6 = 0

def test_case_0():
    var_0 = lambda : (True, 'OK')
    var_1 = 'TestClass'
    var_2 = {}
    var_3 = 0

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not callable'
    var_2 = {var_0: var_1}
    var_3 = 'TestClass'
    var_4 = ()

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = ()

def test_case_0():
    var_0 = lambda : (True, 'OK1')
    var_1 = lambda : (True, 'OK2')
    var_2 = 'TestClass'
    var_3 = {}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_missing_fields_added_when_mandatory_fields_exist. Retrieved 3/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'TestRecord'
    var_4 = module_0.PMap()



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 11/14 statements.
# Partially parsed test_precord_new_without_special_attributes. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/3 statements.
# Partially parsed test_precord_new_with_initial_values_and_kwargs. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'TestRecord'
    var_1 = {}
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = 'b'
    var_8 = (var_7, var_2)
    var_9 = [var_8]
    var_10 = [var_6, var_9]

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2, var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_invariant_exception_raised_when_error_codes_or_missing_fields. Retrieved 5/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = module_0.PMap()
    var_4 = 'error1'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = module_0.serialize()



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.serialize()



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_field_exists_in_precord_fields. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.PMap()
    var_2 = 'key'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_set_with_existing_field. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing_field'
    var_1 = module_0.PMap()
    var_2 = 'existing_field'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_persistent_returns_pmap_when_not_dirty_and_instance_of_cls. Retrieved 10/13 statements.
# Partially parsed test_persistent_returns_new_instance_when_dirty. Retrieved 12/17 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_field_invariant_failures. Retrieved 15/27 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_global_invariant_failures. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'key'
    var_11 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'invariant'
    var_3 = 'factory'
    var_4 = lambda x: x
    var_5 = 'MockClass'
    var_6 = ()
    var_7 = '_precord_fields'
    var_8 = '_precord_mandatory_fields'
    var_9 = '_precord_invariants'
    var_10 = 'field'
    var_11 = set()
    var_12 = []
    var_13 = module_0.PMap()
    var_14 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 22/37 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 17/30 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 22/36 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 19/33 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 15/33 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'MockPMap'
    var_10 = ()
    var_11 = '_buckets'
    var_12 = '_size'
    var_13 = None
    var_14 = 0
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'MockMap'
    var_17 = ()
    var_18 = 'new_buckets'
    var_19 = 1
    var_20 = {var_11: var_18, var_12: var_19}
    var_21 = True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'MockPMap'
    var_10 = ()
    var_11 = '_buckets'
    var_12 = '_size'
    var_13 = 'old_buckets'
    var_14 = 0
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = False

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = {var_6, var_7}
    var_9 = []
    var_10 = {var_2: var_5, var_3: var_8, var_4: var_9}
    var_11 = 'MockPMap'
    var_12 = ()
    var_13 = '_buckets'
    var_14 = '_size'
    var_15 = 'keys'
    var_16 = None
    var_17 = 0
    var_18 = [var_6]
    var_19 = lambda : var_18
    var_20 = {var_13: var_16, var_14: var_17, var_15: var_19}
    var_21 = True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'MockPMap'
    var_10 = ()
    var_11 = '_buckets'
    var_12 = '_size'
    var_13 = None
    var_14 = 0
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'error1'
    var_17 = 'error2'
    var_18 = True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = 'MockPMap'
    var_8 = ()
    var_9 = '_buckets'
    var_10 = '_size'
    var_11 = None
    var_12 = 0
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = True



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_missing_mandatory_fields_are_added_to_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = 'TestRecord'
    var_4 = module_0.PMap()
    var_5 = 'field1'
    var_6 = 'value1'
    var_7 = {var_5: var_6}



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_persistent_with_dirty_state_and_non_cls_instance. Retrieved 12/17 statements.
# Partially parsed test_persistent_with_clean_state_and_cls_instance. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 10/19 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'key'
    var_11 = 'value'

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = {}
    var_10 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()
    var_8 = 'key'
    var_9 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()



# Parsed testcases at query #41
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #42
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_persistent_predicate_false. Retrieved 10/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()



# Parsed testcases at query #44
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 18/24 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 8/13 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 10/15 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 12/17 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 10/15 statements.
# Partially parsed test_precord_new_with_overriding_initial_values. Retrieved 11/16 statements.


def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = '_precord_size'
    var_3 = '_precord_buckets'
    var_4 = 2
    var_5 = 1
    var_6 = 'a'
    var_7 = (var_5, var_6)
    var_8 = [var_7]
    var_9 = 'b'
    var_10 = (var_4, var_9)
    var_11 = [var_10]
    var_12 = [var_8, var_11]
    var_13 = (var_5, var_6)
    var_14 = [var_13]
    var_15 = (var_4, var_9)
    var_16 = [var_15]
    var_17 = [var_14, var_16]

def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = 'a'
    var_3 = 'b'
    var_4 = None
    var_5 = 1
    var_6 = 2
    var_7 = {var_2: var_5, var_3: var_6}

def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = 'a'
    var_3 = 'b'
    var_4 = None
    var_5 = '_factory_fields'
    var_6 = 1
    var_7 = 2
    var_8 = {var_2}
    var_9 = {var_2: var_6, var_3: var_7, var_5: var_8}

def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = 'a'
    var_3 = 'b'
    var_4 = None
    var_5 = 'c'
    var_6 = '_ignore_extra'
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = True
    var_11 = {var_2: var_7, var_3: var_8, var_5: var_9, var_6: var_10}

def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = 'a'
    var_3 = 'b'
    var_4 = None
    var_5 = 1
    var_6 = lambda : var_5
    var_7 = 2
    var_8 = lambda : var_7
    var_9 = {}

def test_case_0():
    var_0 = 'TestPRecord'
    var_1 = {}
    var_2 = 'a'
    var_3 = 'b'
    var_4 = None
    var_5 = 1
    var_6 = lambda : var_5
    var_7 = 2
    var_8 = lambda : var_7
    var_9 = 10
    var_10 = {var_2: var_9}



# Parsed testcases at query #45
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 17/24 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 19/29 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 18/25 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 18/26 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 15/26 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'MockPMap'
    var_10 = ()
    var_11 = '_buckets'
    var_12 = '_size'
    var_13 = 'mock_buckets'
    var_14 = 'mock_size'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'MockPMap'
    var_10 = ()
    var_11 = '_buckets'
    var_12 = '_size'
    var_13 = 'mock_buckets'
    var_14 = 'mock_size'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'MockPM'
    var_17 = {var_11: var_13, var_12: var_14}
    var_18 = False

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'field1'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = 'MockPMap'
    var_11 = ()
    var_12 = '_buckets'
    var_13 = '_size'
    var_14 = 'mock_buckets'
    var_15 = 'mock_size'
    var_16 = {var_12: var_14, var_13: var_15}
    var_17 = True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = 'MockPMap'
    var_10 = ()
    var_11 = '_buckets'
    var_12 = '_size'
    var_13 = 'mock_buckets'
    var_14 = 'mock_size'
    var_15 = {var_11: var_13, var_12: var_14}
    var_16 = 'error1'
    var_17 = True

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = 'MockPMap'
    var_8 = ()
    var_9 = '_buckets'
    var_10 = '_size'
    var_11 = 'mock_buckets'
    var_12 = 'mock_size'
    var_13 = {var_9: var_11, var_10: var_12}
    var_14 = True



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_persistent_with_mandatory_fields_missing. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = {var_0, var_1}
    var_3 = []
    var_4 = 'TestClass'
    var_5 = module_0.PMap()
    var_6 = 'field1'
    var_7 = 'value1'



# Parsed testcases at query #47
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 9/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 5/7 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 1
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.PRecord(**var_7)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'default_value'
    var_2 = 'new_value'
    var_3 = {var_0: var_2}
    var_4 = module_0.PRecord(**var_3)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'default_value'
    var_2 = lambda : var_1
    var_3 = {}
    var_4 = module_0.PRecord(**var_3)



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_extra_fields_not_ignored. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_internal_fields. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : 1
    var_1 = lambda : 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 10
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = 20
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 10/12 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_pickle_support. Retrieved 4/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'extra'
    var_5 = 10
    var_6 = 'test'
    var_7 = 'ignored'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_2]

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
    var_2 = 10
    var_3 = 'test'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 10/12 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_pickle_support. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 'z'
    var_5 = 10
    var_6 = 20
    var_7 = 30
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_2]

def test_case_0():
    var_0 = lambda : 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 10
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = 20
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 9/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/6 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 1
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.PRecord(**var_7)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'initial_value'
    var_2 = 'updated_value'
    var_3 = {var_0: var_2}
    var_4 = module_0.PRecord(**var_3)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = module_0.PRecord()



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 5/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 0
    var_2 = {}
    var_3 = set()
    var_4 = []



# Parsed testcases at query #53
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 7/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/5 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/5 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4, var_0]
    var_6 = module_0.PRecord()

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = [var_0, var_1]
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = module_0.PRecord()

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = True
    var_1 = 'value1'
    var_2 = 'extra_value'
    var_3 = module_0.PRecord()

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'default1'
    var_8 = 'default2'
    var_9 = {var_5: var_7, var_6: var_8}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'default1'
    var_8 = 'default2'
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 'new_value'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = lambda : 'computed1'
    var_8 = 'default2'
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 11/12 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 5/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/5 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 1
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.PRecord(**var_7)
    var_9 = (var_3, var_4)
    var_10 = [var_9]

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'initial_value'
    var_2 = 'new_value'
    var_3 = {var_0: var_2}
    var_4 = module_0.PRecord(**var_3)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = module_0.PRecord()



# Parsed testcases at query #56
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 9/11 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 5/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 8/10 statements.


def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = None
    var_4 = 1
    var_5 = (var_4, var_2)
    var_6 = [var_5]
    var_7 = [var_3, var_6]
    var_8 = {var_0: var_2, var_1: var_7}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = '_factory_fields'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = [var_1]
    var_4 = 1
    var_5 = 2
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}

def test_case_0():
    var_0 = '_ignore_extra'
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = True
    var_5 = 2
    var_6 = 3
    var_7 = {var_0: var_4, var_1: var_4, var_2: var_5, var_3: var_6}



# Parsed testcases at query #57
#--------------------------




def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 10
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_internal_params. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = lambda : 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 'y'
    var_4 = 20
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 10
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = 20
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_persistent_predicate_false. Retrieved 9/17 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_mandatory_fields'
    var_3 = '_precord_invariants'
    var_4 = set()
    var_5 = []
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = module_0.PMap()
    var_8 = False



# Parsed testcases at query #60
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'test_key'
    var_5 = 'test_value'



# Parsed testcases at query #61
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'new_field'
    var_5 = 'value'



# Parsed testcases at query #62
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 2/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 5/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_internal_params. Retrieved 5/8 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'test'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = 10
    var_3 = 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = True

def test_case_0():
    var_0 = 'field1'
    var_1 = 10
    var_2 = {var_0: var_1}
    var_3 = 'field2'
    var_4 = [var_3]

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = 1
    var_1 = 'field1'
    var_2 = 10
    var_3 = (var_1, var_2)
    var_4 = [var_3]



# Parsed testcases at query #63
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_internal_fields. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = lambda : 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 'x'
    var_4 = 5
    var_5 = {var_3: var_4}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 10
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = 20
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]



# Parsed testcases at query #64
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/3 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/6 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_6]
    var_8 = [var_4, var_7]
    var_9 = module_0.PRecord()

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2, var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True



# Parsed testcases at query #65
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 10/14 statements.
# Partially parsed test_precord_new_without_special_attributes. Retrieved 7/12 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_precord_new_with_callable_initial_values. Retrieved 6/11 statements.


def test_case_0():
    var_0 = 5
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = 'b'
    var_5 = 2
    var_6 = (var_4, var_5)
    var_7 = (var_3, var_6)
    var_8 = None
    var_9 = [var_7, var_8, var_8, var_8, var_8]

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2
    var_5 = 10
    var_6 = 20

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = {var_0}
    var_4 = 10
    var_5 = 20

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = True
    var_3 = 10
    var_4 = 20

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = lambda : var_3
    var_5 = 2



# Parsed testcases at query #66
#--------------------------

# Partially parsed test_precord_constructor_with_initial_values. Retrieved 7/8 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/5 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.PRecord(**var_9)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default1'
    var_3 = 'default2'
    var_4 = 'new_value'
    var_5 = {var_0: var_4}
    var_6 = module_0.PRecord(**var_5)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = module_0.PRecord()



# Parsed testcases at query #67
#--------------------------

# Partially parsed test_precord_constructor_with_evolver. Retrieved 4/8 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.PRecord(**var_9)
    var_11 = module_0.PRecord()

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = '_factory_fields'
    var_3 = 1
    var_4 = 2
    var_5 = [var_0]
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = '_ignore_extra'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = True
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.PRecord(**var_8)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 2
    var_1 = module_0.pmap(pre_size=var_0)
    var_2 = None
    var_3 = False



# Parsed testcases at query #68
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 10/12 statements.
# Failed to parse test_precord_constructor_with_callable_initial_values.
# Partially parsed test_precord_constructor_with_existing_instance. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'extra'
    var_5 = 10
    var_6 = 'test'
    var_7 = 'ignored'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'



# Parsed testcases at query #69
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 12/14 statements.
# Partially parsed test_precord_new_without_special_attributes. Retrieved 5/10 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 6/12 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 6/11 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/9 statements.


def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = (var_5,)
    var_7 = 'b'
    var_8 = (var_7, var_2)
    var_9 = (var_8,)
    var_10 = [var_6, var_9]
    var_11 = {var_0: var_2, var_1: var_10}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 1
    var_4 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = 10
    var_4 = 20
    var_5 = lambda : var_4

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = {var_0}
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = None
    var_2 = True
    var_3 = 2



# Parsed testcases at query #70
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 10/12 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 8/10 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_callable_initial. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_existing_instance. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 'test'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 'extra'
    var_5 = 10
    var_6 = 'test'
    var_7 = 'ignored'
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'test'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = [var_2]

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
    var_2 = 10
    var_3 = 'test'



# Parsed testcases at query #71
#--------------------------

# Partially parsed test_persistent_with_clean_pmap_of_correct_type. Retrieved 17/23 statements.


def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = 'MockPMap'
    var_11 = ()
    var_12 = '_buckets'
    var_13 = '_size'
    var_14 = '__class__'
    var_15 = None
    var_16 = 0



# Parsed testcases at query #72
#--------------------------

# Partially parsed test_pr_new_with_special_attributes. Retrieved 6/8 statements.
# Partially parsed test_pr_new_without_special_attributes. Retrieved 2/4 statements.
# Partially parsed test_pr_new_with_factory_fields. Retrieved 4/6 statements.
# Partially parsed test_pr_new_with_ignore_extra. Retrieved 3/5 statements.
# Partially parsed test_pr_new_with_initial_values. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 5
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_1, var_2, var_3, var_4, var_0]

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = [var_2]

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True

def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #73
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()



# Parsed testcases at query #74
#--------------------------

# Partially parsed test_persistent_no_changes. Retrieved 10/13 statements.
# Partially parsed test_persistent_with_changes. Retrieved 12/16 statements.
# Partially parsed test_persistent_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_invariant_failure. Retrieved 8/16 statements.
# Partially parsed test_persistent_global_invariant_failure. Retrieved 8/16 statements.
# Partially parsed test_persistent_field_invariant_failure. Retrieved 20/28 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'key'
    var_11 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'factory'
    var_3 = 'invariant'
    var_4 = lambda x: x
    var_5 = False
    var_6 = 'FIELD_INVARIANT_FAILED'
    var_7 = (var_5, var_6)
    var_8 = lambda x: var_7
    var_9 = {var_2: var_4, var_3: var_8}
    var_10 = 'MockClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = '_precord_mandatory_fields'
    var_14 = '_precord_invariants'
    var_15 = 'field'
    var_16 = set()
    var_17 = []
    var_18 = module_0.PMap()
    var_19 = 'value'



# Parsed testcases at query #75
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'missing1'



# Parsed testcases at query #76
#--------------------------

# Partially parsed test_precord_initial_values_are_used. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #77
#--------------------------

# Partially parsed test_repr_returns_correct_string. Retrieved 4/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 'test'



# Parsed testcases at query #78
#--------------------------

# Partially parsed test_precord_constructor_with_valid_kwargs. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 10/12 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : 1
    var_1 = lambda : 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = 10
    var_5 = 20
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = 10
    var_4 = 20
    var_5 = 30

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = (var_6, var_2)
    var_8 = (var_5, var_7)
    var_9 = (var_8,)



# Parsed testcases at query #79
#--------------------------

# Partially parsed test_precord_metaclass_initialization. Retrieved 1/4 statements.


def test_case_0():
    var_0 = lambda self: True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 5/9 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_factory_fields_and_ignore_extra. Retrieved 5/9 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/3 statements.
# Partially parsed test_precord_new_with_initial_values_and_kwargs. Retrieved 3/5 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 8
    var_3 = var_1 * var_2
    var_4 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = {var_2}
    var_4 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 10
    var_3 = lambda : 20
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 10
    var_3 = lambda : 20
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1
    var_6 = 2



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 3/7 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 4/8 statements.
# Partially parsed test_serialize_empty_record. Retrieved 1/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 123
    var_2 = module_0.serialize()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'value1'
    var_1 = 'value2'
    var_2 = 'custom'
    var_3 = module_0.serialize(var_2)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.serialize()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_instance. Retrieved 12/18 statements.
# Partially parsed test_persistent_with_clean_and_instance. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_invariant_error_codes. Retrieved 15/27 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'test_key'
    var_11 = 'test_value'

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = {}
    var_10 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockField'
    var_1 = ()
    var_2 = 'invariant'
    var_3 = 'factory'
    var_4 = lambda x: x
    var_5 = 'MockClass'
    var_6 = ()
    var_7 = '_precord_fields'
    var_8 = '_precord_mandatory_fields'
    var_9 = '_precord_invariants'
    var_10 = 'test_field'
    var_11 = set()
    var_12 = []
    var_13 = module_0.PMap()
    var_14 = 'test_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_repr_empty.
# Partially parsed test_repr_single_field. Retrieved 2/5 statements.
# Partially parsed test_repr_multiple_fields. Retrieved 4/7 statements.
# Partially parsed test_repr_with_complex_values. Retrieved 8/11 statements.
# Partially parsed test_repr_with_nested_record. Retrieved 3/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 42

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
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = 'a'
    var_7 = {var_6: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_persistent_returns_pmap_when_not_dirty_and_instance_of_cls. Retrieved 10/13 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 12/17 statements.
# Partially parsed test_persistent_creates_new_instance_when_not_instance_of_cls. Retrieved 10/14 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_invariant_errors. Retrieved 11/16 statements.
# Partially parsed test_persistent_raises_invariant_exception_for_missing_fields. Retrieved 11/16 statements.
# Partially parsed test_persistent_checks_global_invariants. Retrieved 14/18 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'key'
    var_11 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'error_code'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'missing_field'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = False
    var_8 = 'global_error'
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = [var_10]
    var_12 = {var_2: var_5, var_3: var_6, var_4: var_11}
    var_13 = module_0.PMap()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'test_key'
    var_5 = 'test_value'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test__new__sets_fields_and_invariants. Retrieved 3/9 statements.
# Partially parsed test__new__inherits_fields_and_invariants. Retrieved 3/9 statements.
# Failed to parse test__new__sets_mandatory_fields.
# Partially parsed test__new__sets_initial_values. Retrieved 1/4 statements.
# Partially parsed test__new__sets_empty_slots. Retrieved 1/3 statements.
# Partially parsed test__new__raises_type_error_for_non_callable_invariant. Retrieved 1/4 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = lambda self: True
    var_2 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = lambda self: True
    var_2 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

def test_case_0():
    var_0 = 'not callable'



# Parsed testcases at query #8
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_new_with_no_bases_and_empty_dct. Retrieved 4/5 statements.
# Partially parsed test_new_with_single_base_and_inherited_fields. Retrieved 2/8 statements.
# Partially parsed test_new_with_multiple_bases_and_inherited_fields. Retrieved 2/10 statements.
# Partially parsed test_new_with_invariant_function. Retrieved 7/4 statements.
# Partially parsed test_new_with_invalid_invariant. Retrieved 5/7 statements.
# Partially parsed test_new_with_field_in_dct. Retrieved 6/7 statements.
# Partially parsed test_new_with_field_and_initial_value. Retrieved 7/8 statements.
# Partially parsed test_new_with_no_initial_value_field. Retrieved 6/7 statements.


def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = {}
    var_3 = set()

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}

def test_case_0():
    var_0 = 'TestClass'
    var_1 = {}

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = 'TestClass'
    var_4 = ()
    var_5 = '__invariant__'
    var_6 = set()

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = 'TestClass'
    var_4 = ()
    var_5 = '__invariant__'
    var_6 = set()

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '__invariant__'
    var_3 = 'not_callable'
    var_4 = {var_2: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'x'
    var_3 = True
    var_4 = module_0._PField(var_3)
    var_5 = {var_2: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'x'
    var_3 = 42
    var_4 = module_0._PField(var_3)
    var_5 = {var_2: var_4}
    var_6 = set()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = 'x'
    var_3 = module_0._PField()
    var_4 = {var_2: var_3}
    var_5 = set()



# Parsed testcases at query #10
#--------------------------

# Failed to parse test__new__sets_fields_correctly.
# Partially parsed test__new__sets_mandatory_fields. Retrieved 1/3 statements.
# Partially parsed test__new__sets_initial_values. Retrieved 1/3 statements.
# Partially parsed test__new__stores_invariants. Retrieved 6/4 statements.
# Failed to parse test__new__inherits_invariants.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

def test_case_0():
    pass

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = '_precord_invariants'
    var_4 = len(var_2)
    assert var_4 == 1
    var_5 = 0

def test_case_0():
    var_0 = True
    var_1 = 'OK'
    var_2 = (var_0, var_1)
    var_3 = '_precord_invariants'
    var_4 = len(var_2)
    assert var_4 == 1
    var_5 = 0

def test_case_0():
    var_0 = 'not callable'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_with_dirty_and_not_instance. Retrieved 3/7 statements.
# Partially parsed test_persistent_without_dirty_and_is_instance. Retrieved 4/7 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 4/9 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 6/11 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'value1'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = module_0.PMap()
    var_3 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'error_code'
    var_2 = (var_0, var_1)
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 'invalid_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'global_error'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.PMap()
    var_5 = 'field1'
    var_6 = 'value1'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 16/24 statements.
# Partially parsed test_set_with_invalid_type. Retrieved 17/26 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 16/24 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 8/12 statements.
# Partially parsed test_set_with_ignore_extra_true_and_compliant_factory. Retrieved 16/24 statements.
# Partially parsed test_set_with_ignore_extra_false_and_compliant_factory. Retrieved 17/25 statements.
# Partially parsed test_set_with_factory_field_not_in_factory_fields. Retrieved 17/25 statements.
# Partially parsed test_set_with_factory_field_in_factory_fields. Retrieved 17/26 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = 'field1'
    var_14 = module_0.PMap()
    var_15 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = 'field1'
    var_14 = module_0.PMap()
    var_15 = 'field1'
    var_16 = 'not_an_int'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = False
    var_7 = 'ERROR'
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = 'field1'
    var_14 = module_0.PMap()
    var_15 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'TestClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = {}
    var_4 = {var_2: var_3}
    var_5 = module_0.PMap()
    var_6 = 'nonexistent'
    var_7 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x, ignore_extra=False: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = 'field1'
    var_14 = module_0.PMap()
    var_15 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x, ignore_extra=False: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = 'field1'
    var_14 = module_0.PMap()
    var_15 = False
    var_16 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = lambda x: x
    var_6 = True
    var_7 = None
    var_8 = (var_6, var_7)
    var_9 = lambda x: var_8
    var_10 = 'TestClass'
    var_11 = ()
    var_12 = '_precord_fields'
    var_13 = 'field1'
    var_14 = module_0.PMap()
    var_15 = []
    var_16 = 10

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'type'
    var_3 = 'factory'
    var_4 = 'invariant'
    var_5 = 1
    var_6 = lambda x: x + var_5
    var_7 = True
    var_8 = None
    var_9 = (var_7, var_8)
    var_10 = lambda x: var_9
    var_11 = 'TestClass'
    var_12 = ()
    var_13 = '_precord_fields'
    var_14 = 'field1'
    var_15 = module_0.PMap()
    var_16 = 10



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_predicate_at_line_5_evaluates_to_false. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test__new_creates_class_with_correct_attributes. Retrieved 6/15 statements.


def test_case_0():
    var_0 = lambda self: True
    var_1 = '_precord_fields'
    var_2 = '_precord_invariants'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_initial_values'
    var_5 = '__slots__'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_repr_format. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = 'TestRecord('
    var_3 = ')'



# Parsed testcases at query #17
#--------------------------

# Failed to parse test__new__sets_precord_mandatory_fields.
# Partially parsed test__new__sets_precord_initial_values. Retrieved 1/3 statements.
# Partially parsed test__new__inherits_invariants. Retrieved 3/17 statements.
# Partially parsed test__new__wraps_invariants. Retrieved 8/7 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()

def test_case_0():
    pass

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0._PField()
    var_1 = module_0._PField()

def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = 1

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = ()
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = 0
    var_7 = None

def test_case_0():
    var_0 = True
    var_1 = ()
    var_2 = (var_0, var_1)
    var_3 = ()
    var_4 = (var_0, var_3)
    var_5 = [var_2, var_4]
    var_6 = 0
    var_7 = None

def test_case_0():
    var_0 = 'not callable'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_repr. Retrieved 4/7 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = 2
    var_3 = 'updated'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_initial_values_are_used. Retrieved 2/4 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 1
    var_8 = 2
    var_9 = {var_5: var_7, var_6: var_8}



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_predicate_line_1.




# Parsed testcases at query #21
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields_exist. Retrieved 13/19 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = '__name__'
    var_6 = {}
    var_7 = set()
    var_8 = []
    var_9 = {var_2: var_6, var_3: var_7, var_4: var_8, var_5: var_0}
    var_10 = module_0.PMap()
    var_11 = 'error1'
    var_12 = 'field1'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_repr_format. Retrieved 4/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 'test'
    var_2 = 'TestRecord('
    var_3 = ')'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_persistent_returns_same_instance_when_not_dirty_and_correct_type. Retrieved 3/7 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 5/10 statements.
# Partially parsed test_persistent_creates_new_instance_when_not_correct_type. Retrieved 3/7 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_mandatory_fields_missing. Retrieved 5/10 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_invariant_error_codes_exist. Retrieved 5/10 statements.
# Partially parsed test_persistent_raises_invariant_exception_when_global_invariants_fail. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = None
    var_2 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = None
    var_2 = False
    var_3 = 'field'
    var_4 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = None
    var_2 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = module_0.pmap()
    var_3 = None
    var_4 = False

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.pmap()
    var_1 = None
    var_2 = False
    var_3 = 'error1'
    var_4 = 'error2'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = False
    var_1 = 'global_error'
    var_2 = (var_0, var_1)
    var_3 = lambda x: var_2
    var_4 = module_0.pmap()
    var_5 = None



# Parsed testcases at query #24
#--------------------------

# Failed to parse test_precord_new_without_special_attributes.




# Parsed testcases at query #25
#--------------------------

# Failed to parse test_predicate_false.




# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 5/8 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'value1'
    var_3 = 'value2'
    var_4 = module_0.serialize()



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 4/9 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_returns_dict. Retrieved 1/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.serialize()



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 7/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = set()
    var_1 = []
    var_2 = {}
    var_3 = 'TestClass'
    var_4 = module_0.PMap()
    var_5 = 'test_key'
    var_6 = 'test_value'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_precord_new_with_precord_size_and_buckets. Retrieved 7/10 statements.
# Partially parsed test_precord_new_without_precord_size_and_buckets. Retrieved 4/7 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/10 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 6/9 statements.
# Failed to parse test_precord_new_with_initial_values.
# Partially parsed test_precord_new_with_initial_values_and_kwargs. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 2
    var_1 = None
    var_2 = 1
    var_3 = 'a'
    var_4 = (var_2, var_3)
    var_5 = [var_4]
    var_6 = [var_1, var_5]

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
    var_2 = 'field1'
    var_3 = 1
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = True

def test_case_0():
    var_0 = 100



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_persistent_with_mandatory_fields_missing. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = module_0.PMap()
    var_6 = 'field1'
    var_7 = 'value1'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 5/11 statements.
# Partially parsed test_set_with_invalid_type. Retrieved 5/12 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 5/11 statements.
# Partially parsed test_set_with_nonexistent_field. Retrieved 4/8 statements.
# Partially parsed test_set_with_factory_field_not_in_factory_fields. Retrieved 6/12 statements.
# Partially parsed test_set_with_ignore_extra_and_compliant_factory. Retrieved 8/14 statements.
# Partially parsed test_set_with_ignore_extra_and_non_compliant_factory. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x * 2
    var_1 = lambda x: (True, None)
    var_2 = 'test_field'
    var_3 = module_0.PMap()
    var_4 = 'test_field'
    var_5 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: (True, None)
    var_2 = 'test_field'
    var_3 = module_0.PMap()
    var_4 = 'test_field'
    var_5 = 'not_an_int'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: (False, 'test_error') if x < 0 else (True, None)
    var_2 = 'test_field'
    var_3 = module_0.PMap()
    var_4 = 'test_field'
    var_5 = -1

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.PMap()
    var_2 = 'nonexistent_field'
    var_3 = 123

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x * 2
    var_1 = lambda x: (True, None)
    var_2 = 'test_field'
    var_3 = module_0.PMap()
    var_4 = []
    var_5 = 'test_field'
    var_6 = 5

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x, ignore_extra=False: x
    var_1 = lambda x: (True, None)
    var_2 = 'test_field'
    var_3 = module_0.PMap()
    var_4 = True
    var_5 = 'test_field'
    var_6 = 'key'
    var_7 = 'value'
    var_8 = {var_6: var_7}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda x: (True, None)
    var_2 = 'test_field'
    var_3 = module_0.PMap()
    var_4 = True
    var_5 = 'test_field'
    var_6 = 5



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_precord_new_without_special_attributes. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_new_with_initial_values_and_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 5/9 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/7 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 2
    var_1 = 'a'
    var_2 = 1
    var_3 = (var_1, var_2)
    var_4 = [var_3]
    var_5 = 'b'
    var_6 = (var_5, var_0)
    var_7 = [var_6]
    var_8 = [var_4, var_7]
    var_9 = module_0.PRecord()
    var_10 = len(var_9)
    assert var_10 == 2

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = 2
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 3
    var_6 = 4

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'a'
    var_3 = 'b'
    var_4 = {var_2, var_3}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_persistent_raises_invariant_exception_when_error_codes_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'field1'



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_instance. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_clean_and_instance. Retrieved 1/5 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 1/4 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 3/7 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'value1'

def test_case_0():
    var_0 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field_with_invariant'
    var_2 = 'invalid_value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = module_0.PMap()
    var_1 = 'field1'
    var_2 = 'value1'
    var_3 = 'field2'
    var_4 = 'value2'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_precord_initial_values_are_used. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #37
#--------------------------

# Partially parsed test_precord_initial_values_used. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 1
    var_3 = lambda : 2
    var_4 = {var_0: var_2, var_1: var_3}



# Parsed testcases at query #38
#--------------------------

# Partially parsed test_invariant_exception_raised_when_error_codes_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'field1'



# Parsed testcases at query #39
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_internal_params. Retrieved 8/10 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : 1
    var_1 = lambda : 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 100
    var_4 = {var_2: var_3}
    var_5 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 'y'
    var_5 = 10
    var_6 = 20
    var_7 = {var_3: var_5, var_4: var_6}



# Parsed testcases at query #40
#--------------------------

# Partially parsed test_persistent_with_mandatory_fields_missing. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = 'TestRecord'
    var_6 = module_0.PMap()



# Parsed testcases at query #41
#--------------------------

# Partially parsed test_persistent_with_dirty_and_valid_fields. Retrieved 6/11 statements.
# Partially parsed test_persistent_with_clean_and_valid_fields. Retrieved 6/11 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 4/8 statements.
# Partially parsed test_persistent_with_invariant_error. Retrieved 5/11 statements.
# Partially parsed test_persistent_with_global_invariant_error. Retrieved 4/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 'value1'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = {var_3: var_4}

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = {var_1}
    var_3 = []
    var_4 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'field1'
    var_5 = 'value1'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = module_0.PMap()



# Parsed testcases at query #42
#--------------------------

# Partially parsed test_persistent_raises_when_invariant_errors_or_missing_fields. Retrieved 6/12 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'field1'



# Parsed testcases at query #43
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_cls_instance. Retrieved 12/17 statements.
# Partially parsed test_persistent_with_clean_and_cls_instance. Retrieved 13/17 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 8/16 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'key'
    var_11 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = var_9._buckets
    var_11 = module_0.PMap()
    var_12 = var_11._size

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()



# Parsed testcases at query #44
#--------------------------

# Failed to parse test_predicate_line_1_false.




# Parsed testcases at query #45
#--------------------------

# Partially parsed test_field_exists_in_precord_fields. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = module_0.PMap()
    var_2 = 'key'



# Parsed testcases at query #46
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 2/7 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 1/5 statements.
# Partially parsed test_precord_constructor_with_callable_default. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/8 statements.
# Partially parsed test_precord_constructor_with_internal_fields. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 10
    var_1 = 'hello'

def test_case_0():
    var_0 = 10

def test_case_0():
    var_0 = lambda : 'default'
    var_1 = 10

def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = 'extra'
    var_3 = True

def test_case_0():
    var_0 = 10
    var_1 = 'y'
    var_2 = 'default'
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 2
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = 'hello'
    var_5 = {var_1: var_3, var_2: var_4}



# Parsed testcases at query #47
#--------------------------




def test_case_0():
    pass



# Parsed testcases at query #48
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 12/13 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 10/11 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 9/11 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'field1'
    var_4 = 'value1'
    var_5 = (var_3, var_4)
    var_6 = 'field2'
    var_7 = 'value2'
    var_8 = (var_6, var_7)
    var_9 = [var_5, var_8]
    var_10 = {var_0: var_2, var_1: var_9}
    var_11 = module_0.PRecord(**var_10)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = '_factory_fields'
    var_3 = 'value1'
    var_4 = 'value2'
    var_5 = [var_0]
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'extra_field'
    var_3 = '_ignore_extra'
    var_4 = 'value1'
    var_5 = 'value2'
    var_6 = 'extra_value'
    var_7 = True
    var_8 = {var_0: var_4, var_1: var_5, var_2: var_6, var_3: var_7}
    var_9 = module_0.PRecord(**var_8)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'initial_value1'
    var_3 = 'initial_value2'
    var_4 = 'updated_value1'
    var_5 = {var_0: var_4}
    var_6 = module_0.PRecord(**var_5)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'initial_value1'
    var_3 = lambda : var_2
    var_4 = 'initial_value2'
    var_5 = lambda : var_4
    var_6 = 'updated_value1'
    var_7 = {var_0: var_6}
    var_8 = module_0.PRecord(**var_7)



# Parsed testcases at query #49
#--------------------------

# Partially parsed test_persistent_raises_when_invariant_error_codes_or_missing_fields. Retrieved 6/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = ()
    var_3 = module_0.PMap()
    var_4 = 'error1'
    var_5 = 'field1'



# Parsed testcases at query #50
#--------------------------

# Partially parsed test_field_exists_in_precord_fields. Retrieved 2/8 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'existing_field'
    var_1 = module_0.PMap()
    var_2 = 'existing_field'



# Parsed testcases at query #51
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : 1
    var_1 = lambda : 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 'x'
    var_3 = 'y'
    var_4 = [var_2, var_3]
    var_5 = 10
    var_6 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = True
    var_3 = 10
    var_4 = 20
    var_5 = 30

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]



# Parsed testcases at query #52
#--------------------------

# Partially parsed test_persistent_with_dirty_and_non_instance. Retrieved 12/18 statements.
# Partially parsed test_persistent_with_clean_and_instance. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_missing_mandatory_fields. Retrieved 11/15 statements.
# Partially parsed test_persistent_with_invariant_errors. Retrieved 8/16 statements.
# Partially parsed test_persistent_with_global_invariant_failure. Retrieved 8/16 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = module_0.PMap()
    var_10 = 'key'
    var_11 = 'value'

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = []
    var_8 = {var_2: var_5, var_3: var_6, var_4: var_7}
    var_9 = {}
    var_10 = 0

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = 'mandatory_field'
    var_7 = {var_6}
    var_8 = []
    var_9 = {var_2: var_5, var_3: var_7, var_4: var_8}
    var_10 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'MockClass'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_mandatory_fields'
    var_4 = '_precord_invariants'
    var_5 = {}
    var_6 = set()
    var_7 = module_0.PMap()



# Parsed testcases at query #53
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 11/12 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/6 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 2
    var_3 = 'a'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'b'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]
    var_9 = {var_0: var_2, var_1: var_8}
    var_10 = module_0.PRecord(**var_9)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'default1'
    var_3 = 'default2'
    var_4 = 'new_value'
    var_5 = {var_0: var_4}
    var_6 = module_0.PRecord(**var_5)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = module_0.PRecord()



# Parsed testcases at query #54
#--------------------------

# Partially parsed test_precord_mandatory_fields_are_checked. Retrieved 6/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {var_1, var_2}
    var_4 = []
    var_5 = module_0.PMap()
    var_6 = 'field1'
    var_7 = 'value1'



# Parsed testcases at query #55
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 9/10 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/7 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 8/9 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 4/6 statements.


import pyrsistent._precord as module_0

def test_case_0():
    var_0 = '_precord_size'
    var_1 = '_precord_buckets'
    var_2 = 10
    var_3 = 'key'
    var_4 = 'value'
    var_5 = (var_3, var_4)
    var_6 = [var_5]
    var_7 = {var_0: var_2, var_1: var_6}
    var_8 = module_0.PRecord(**var_7)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = '_factory_fields'
    var_2 = 'value1'
    var_3 = [var_0]
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = module_0.PRecord(**var_4)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'extra_field'
    var_2 = '_ignore_extra'
    var_3 = 'value1'
    var_4 = 'extra_value'
    var_5 = True
    var_6 = {var_0: var_3, var_1: var_4, var_2: var_5}
    var_7 = module_0.PRecord(**var_6)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'initial_value'
    var_2 = 'new_value'
    var_3 = {var_0: var_2}
    var_4 = module_0.PRecord(**var_3)

import pyrsistent._precord as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'computed_value'
    var_2 = lambda : var_1
    var_3 = module_0.PRecord()



# Parsed testcases at query #56
#--------------------------

# Failed to parse test_predicate_line_1_false.




# Parsed testcases at query #57
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 6/11 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'test_key'
    var_5 = 'test_value'



# Parsed testcases at query #58
#--------------------------

# Partially parsed test_persistent_creates_new_instance_when_dirty_or_not_instance. Retrieved 5/10 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = module_0.PMap()



# Parsed testcases at query #59
#--------------------------

# Partially parsed test_precord_constructor_with_valid_fields. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_default_values. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_callable_defaults. Retrieved 2/4 statements.
# Partially parsed test_precord_constructor_with_extra_fields_ignored. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 7/9 statements.
# Partially parsed test_precord_constructor_with_precord_size_and_buckets. Retrieved 9/11 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20

def test_case_0():
    var_0 = 1
    var_1 = 2

def test_case_0():
    var_0 = lambda : 1
    var_1 = lambda : 2

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 30
    var_5 = True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = 'x'
    var_5 = 'y'
    var_6 = {var_4, var_5}

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 2
    var_3 = 'x'
    var_4 = 1
    var_5 = (var_3, var_4)
    var_6 = 'y'
    var_7 = (var_6, var_2)
    var_8 = [var_5, var_7]



