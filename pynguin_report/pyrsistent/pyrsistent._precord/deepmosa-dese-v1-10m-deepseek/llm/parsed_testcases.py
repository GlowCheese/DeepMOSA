####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_pmap_when_not_dirty_and_already_instance. Retrieved 3/7 statements.
# Partially parsed test_persistent_creates_new_instance_when_dirty. Retrieved 6/15 statements.


import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = module_0.PMap()

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'field'
    var_5 = 'value'

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = module_0.PMap()

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

import pyrsistent._pmap as module_0

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = module_0.PMap()
    var_4 = 'new_field'
    var_5 = 'new_value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 1/4 statements.
# Partially parsed test_precord_new_with_kwargs_overrides_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 2/5 statements.
# Partially parsed test_precord_new_without_ignore_extra_raises_attribute_error. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_invariant_failure. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_missing_mandatory_fields. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_valid_data. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_checked_type_factory. Retrieved 1/7 statements.
# Partially parsed test_precord_new_with_callable_initial_value. Retrieved 1/4 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 20

def test_case_0():
    var_0 = 'field1'
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = 5

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 2

def test_case_0():
    var_0 = 'field1'
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 'field1'
    var_1 = -1

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = {var_2}
    var_4 = 'test'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 100
    var_3 = 'hello'

def test_case_0():
    var_0 = 'inner_field'
    var_1 = 'field1'
    var_2 = 5

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field1'
    var_2 = lambda : 42
    var_3 = {var_1: var_2}



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_repr_with_empty_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_special_characters_in_field_value. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_integer_field_name_and_value. Retrieved 4/8 statements.


def test_case_0():
    var_0 = ()
    var_1 = 'name'
    var_2 = 'Alice'

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = 10
    var_4 = 'test'

def test_case_0():
    var_0 = ()
    var_1 = {}

def test_case_0():
    var_0 = ()
    var_1 = 'data'
    var_2 = 'key'
    var_3 = 'value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = ()
    var_1 = 'text'
    var_2 = 'line1\nline2'

def test_case_0():
    var_0 = ()
    var_1 = '123'
    var_2 = '123'
    var_3 = 456
    var_4 = {var_2: var_3}



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 5/9 statements.
# Partially parsed test_precord_repr_with_no_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 7/11 statements.
# Partially parsed test_precord_repr_with_special_characters_in_field_value. Retrieved 4/8 statements.


def test_case_0():
    var_0 = ()
    var_1 = 'name'
    var_2 = {}
    var_3 = 'Alice'
    var_4 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = {}
    var_4 = 10
    var_5 = 'test'
    var_6 = "TestRecord(x=10, y='test')"

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}
    var_3 = 'TestRecord()'

def test_case_0():
    var_0 = ()
    var_1 = 'data'
    var_2 = 'count'
    var_3 = {}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 5
    var_8 = "TestRecord(data={'key': 'value'}, count=5)"

def test_case_0():
    var_0 = ()
    var_1 = 'text'
    var_2 = {}
    var_3 = 'line1\nline2'
    var_4 = "TestRecord(text='line1\\nline2')"



# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 3/15 statements.
# Partially parsed test_set_with_field_factory_exception. Retrieved 3/19 statements.
# Partially parsed test_set_with_field_invariant_failure. Retrieved 3/15 statements.
# Partially parsed test_set_with_invalid_field. Retrieved 4/11 statements.
# Partially parsed test_set_with_factory_fields_skipped. Retrieved 4/16 statements.
# Partially parsed test_set_with_ignore_extra_complaint. Retrieved 4/16 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'invalid'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = set()
    var_3 = 'key'
    var_4 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = True
    var_3 = 'key'
    var_4 = 5



# Parsed testcases at query #9
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 16/41 statements.
# Partially parsed test___new___handles_no_invariants. Retrieved 10/31 statements.
# Partially parsed test___new___raises_on_non_callable_invariant. Retrieved 7/21 statements.
# Partially parsed test___new___merges_inherited_fields. Retrieved 11/34 statements.


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = lambda self: (True, ())
    var_2 = 'base1_field'
    var_3 = lambda self: (True, ())
    var_4 = 'base2_field'
    var_5 = 'new_field'
    var_6 = False
    var_7 = 'new_initial'
    var_8 = module_1._PField(var_6, var_7)
    var_9 = {var_5: var_8}
    var_10 = '_precord_fields'
    var_11 = '_precord_invariants'
    var_12 = '__invariant__'
    var_13 = var_9[var_10]
    var_14 = var_9[var_10]
    var_15 = var_9[var_11]
    var_16 = len(var_15)
    assert var_16 == 2
    var_17 = var_9[var_11]

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'field'
    var_2 = {}
    var_3 = '_precord_fields'
    var_4 = '_precord_invariants'
    var_5 = '__invariant__'
    var_6 = var_2[var_3]
    var_7 = var_2[var_3]
    var_8 = set()
    var_9 = var_2[var_4]
    var_10 = len(var_9)
    assert var_10 == 0

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'not callable'
    var_2 = {}
    var_3 = {}
    var_4 = '_precord_fields'
    var_5 = '_precord_invariants'
    var_6 = '__invariant__'

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'grand_field'
    var_2 = 'parent_field'
    var_3 = 'child_field'
    var_4 = False
    var_5 = 'child_initial'
    var_6 = module_1._PField(var_4, var_5)
    var_7 = {var_3: var_6}
    var_8 = '_precord_fields'
    var_9 = '_precord_invariants'
    var_10 = '__invariant__'
    var_11 = var_7[var_8]
    var_12 = var_7[var_8]



# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_class. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda s: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 10/32 statements.
# Partially parsed test___new___handles_invariants. Retrieved 6/22 statements.
# Partially parsed test___new___raises_on_non_callable_invariant. Retrieved 5/10 statements.
# Partially parsed test___new___sets_slots. Retrieved 7/8 statements.
# Partially parsed test___new___merges_invariant_results. Retrieved 6/14 statements.


import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'base1_field'
    var_2 = 'base2_field'
    var_3 = 'custom_field'
    var_4 = True
    var_5 = module_1._PField(var_0, var_4)
    var_6 = {var_3: var_5}
    var_7 = '_precord_fields'
    var_8 = '_precord_invariants'
    var_9 = '__invariant__'
    var_10 = var_6[var_7]
    var_11 = var_6[var_7]

def test_case_0():
    var_0 = '__invariant__'
    var_1 = '_precord_fields'
    var_2 = '_precord_invariants'
    var_3 = 0
    var_4 = None
    var_5 = 1

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = '__invariant__'

import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = '_precord_invariants'
    var_5 = '__invariant__'
    var_6 = module_1.store_invariants(var_0, var_1, var_4, var_5)

def test_case_0():
    var_0 = '__invariant__'
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = '_precord_invariants'
    var_4 = 0
    var_5 = None

import pyrsistent._field_common as module_0
import pyrsistent._checked_types as module_1

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = '_precord_fields'
    var_3 = module_0.set_fields(var_0, var_1, var_2)
    var_4 = '_precord_invariants'
    var_5 = '__invariant__'
    var_6 = module_1.store_invariants(var_0, var_1, var_4, var_5)



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_store_invariants_wraps_invariants. Retrieved 11/17 statements.
# Partially parsed test_store_invariants_merges_multiple_results. Retrieved 7/14 statements.
# Partially parsed test_store_invariants_raises_on_non_callable. Retrieved 4/8 statements.
# Partially parsed test_store_invariants_inherits_from_bases. Retrieved 11/19 statements.
# Partially parsed test_store_invariants_includes_current_dict. Retrieved 4/12 statements.
# Partially parsed test_store_invariants_handles_duplicate_inheritance. Retrieved 8/18 statements.


def test_case_0():
    var_0 = lambda *args, **kwargs: (True, ())
    var_1 = lambda *args, **kwargs: (False, ('error',))
    var_2 = {}
    var_3 = '_precord_invariants'
    var_4 = '__invariant__'
    var_5 = var_2[var_3]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = 1
    var_10 = var_5[var_9]

def test_case_0():
    var_0 = {}
    var_1 = '_precord_invariants'
    var_2 = '__invariant__'
    var_3 = var_0[var_1]
    var_4 = len(var_3)
    assert var_4 == 1
    var_5 = 0
    var_6 = var_3[var_5]

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = '_precord_invariants'
    var_3 = '__invariant__'

def test_case_0():
    var_0 = lambda *args, **kwargs: (True, ())
    var_1 = lambda *args, **kwargs: (False, ('base2',))
    var_2 = {}
    var_3 = '_precord_invariants'
    var_4 = '__invariant__'
    var_5 = var_2[var_3]
    var_6 = len(var_5)
    assert var_6 == 2
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = 1
    var_10 = var_5[var_9]

def test_case_0():
    var_0 = ()
    var_1 = '__invariant__'
    var_2 = '_precord_invariants'
    var_3 = 0

def test_case_0():
    var_0 = lambda *args, **kwargs: (True, ())
    var_1 = {}
    var_2 = '_precord_invariants'
    var_3 = '__invariant__'
    var_4 = var_1[var_2]
    var_5 = len(var_4)
    assert var_5 == 1
    var_6 = 0
    var_7 = var_4[var_6]



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_new_creates_instance_with_special_attributes. Retrieved 3/11 statements.
# Partially parsed test_precord_new_uses_evolver_for_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_applies_initial_values_from_class. Retrieved 2/5 statements.
# Partially parsed test_precord_new_overrides_initial_values_with_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_new_handles_factory_fields_parameter. Retrieved 4/9 statements.
# Partially parsed test_precord_new_handles_ignore_extra_parameter. Retrieved 3/6 statements.
# Partially parsed test_precord_new_raises_attribute_error_for_unknown_field. Retrieved 2/6 statements.
# Partially parsed test_precord_new_invokes_field_factory. Retrieved 2/5 statements.
# Partially parsed test_precord_new_validates_field_type. Retrieved 2/6 statements.
# Partially parsed test_precord_new_enforces_field_invariants. Retrieved 2/6 statements.
# Partially parsed test_precord_new_enforces_mandatory_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_new_enforces_global_invariants. Retrieved 4/8 statements.
# Partially parsed test_precord_new_returns_same_instance_if_no_changes. Retrieved 2/8 statements.
# Partially parsed test_precord_new_with_factory_fields_none_uses_original_value. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_ignore_extra_false_raises_for_extra_fields. Retrieved 4/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {}
    var_4 = 42
    var_5 = 'test'

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = lambda : 100
    var_6 = 'default'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = lambda : 100
    var_6 = 'default'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 200

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 2
    var_3 = lambda x: x * var_2
    var_4 = 21

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = True
    var_3 = 2

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 1

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 5

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'not_an_int'

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = -1

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = {var_3}
    var_5 = 'test'

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = lambda r: (r['field1'] + r['field2'] == 10, 'ERR_SUM')
    var_4 = [var_3]
    var_5 = 3
    var_6 = 4

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 1

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = None
    var_3 = 21

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = False
    var_3 = 1
    var_4 = 2



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_with_no_serializers. Retrieved 7/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 7/12 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 8/13 statements.
# Partially parsed test_serialize_with_multiple_fields_and_serializers. Retrieved 10/17 statements.
# Partially parsed test_serialize_empty_record. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'value1'
    var_6 = 42
    var_7 = module_0.serialize()
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = {var_8: var_5, var_9: var_6}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = module_0.serialize()
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'custom_value1'
    var_8 = {var_5: var_7, var_6: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 42
    var_4 = 'fmt'
    var_5 = module_0.serialize(var_4)
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = 'fmt_value1'
    var_9 = {var_6: var_8, var_7: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = 5
    var_4 = 20
    var_5 = 'test'
    var_6 = module_0.serialize()
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field3'
    var_10 = 10
    var_11 = 30
    var_12 = {var_7: var_10, var_8: var_11, var_9: var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.serialize()
    var_2 = {}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 5/9 statements.
# Partially parsed test_precord_repr_with_no_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 7/11 statements.
# Partially parsed test_precord_repr_with_special_characters_in_string. Retrieved 4/8 statements.


def test_case_0():
    var_0 = ()
    var_1 = 'name'
    var_2 = {}
    var_3 = 'Alice'
    var_4 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = {}
    var_4 = 10
    var_5 = 'test'
    var_6 = "TestRecord(x=10, y='test')"

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}
    var_3 = 'TestRecord()'

def test_case_0():
    var_0 = ()
    var_1 = 'data'
    var_2 = 'count'
    var_3 = {}
    var_4 = 'key'
    var_5 = 'value'
    var_6 = {var_4: var_5}
    var_7 = 5
    var_8 = "TestRecord(data={'key': 'value'}, count=5)"

def test_case_0():
    var_0 = ()
    var_1 = 'text'
    var_2 = {}
    var_3 = 'line1\nline2'
    var_4 = "TestRecord(text='line1\\nline2')"



# Parsed testcases at query #17
#--------------------------

# Partially parsed test___new___creates_record_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test___new___creates_record_with_initial_values. Retrieved 2/4 statements.
# Partially parsed test___new___uses_initial_values_from_class. Retrieved 2/4 statements.
# Partially parsed test___new___overrides_class_initial_values_with_kwargs. Retrieved 3/5 statements.
# Partially parsed test___new___handles_callable_initial_values. Retrieved 2/4 statements.
# Partially parsed test___new___raises_attribute_error_for_unknown_field. Retrieved 2/5 statements.
# Partially parsed test___new___applies_factory_fields. Retrieved 3/6 statements.
# Partially parsed test___new___applies_ignore_extra. Retrieved 4/6 statements.
# Partially parsed test___new___raises_invariant_exception_for_missing_mandatory_fields. Retrieved 2/6 statements.
# Partially parsed test___new___raises_invariant_exception_for_field_invariant. Retrieved 1/5 statements.
# Partially parsed test___new___raises_invariant_exception_for_global_invariant. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 5
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = 5
    var_5 = {var_3: var_4}
    var_6 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'x'
    var_4 = lambda : 7
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 10

def test_case_0():
    var_0 = 'x'
    var_1 = 'x'
    var_2 = {var_1}
    var_3 = '10'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 10
    var_5 = 20

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 'x'
    var_3 = {var_2}
    var_4 = 10

def test_case_0():
    var_0 = 'x'
    var_1 = -5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = -5
    var_6 = -5



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 3/7 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_callable_initial_values. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_overrides_initial_values. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_empty. Retrieved 2/5 statements.


def test_case_0():
    var_0 = ()
    var_1 = 0
    var_2 = ()

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'test'
    var_4 = 42

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = 'factory_value'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'value1'
    var_3 = 'extra'
    var_4 = True

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = lambda : 'default'
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = lambda : 'default'
    var_4 = {var_2: var_3}
    var_5 = 'override'

def test_case_0():
    var_0 = ()
    var_1 = {}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 5/14 statements.
# Partially parsed test_set_with_factory_exception. Retrieved 4/15 statements.
# Partially parsed test_set_with_type_check_failure. Retrieved 6/16 statements.
# Partially parsed test_set_with_invariant_failure. Retrieved 5/14 statements.
# Partially parsed test_set_with_ignore_extra_compliant_factory. Retrieved 5/18 statements.
# Partially parsed test_set_with_non_existent_field. Retrieved 5/12 statements.
# Partially parsed test_set_with_factory_fields_skipped. Retrieved 6/15 statements.


def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda self, value: (True, None)
    var_2 = 'key'
    var_3 = {}
    var_4 = 'key'
    var_5 = 5

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = 'key'
    var_2 = {}
    var_3 = 'key'
    var_4 = 5

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda self, value: (True, None)
    var_2 = 'key'
    var_3 = 'MockDestinationCls'
    var_4 = {}
    var_5 = 'key'
    var_6 = 'string'

def test_case_0():
    var_0 = lambda x: x
    var_1 = lambda self, value: (False, 'error') if value < 0 else (True, None)
    var_2 = 'key'
    var_3 = {}
    var_4 = 'key'
    var_5 = -1

def test_case_0():
    var_0 = lambda self, value: (True, None)
    var_1 = 'key'
    var_2 = {}
    var_3 = True
    var_4 = 'key'
    var_5 = 5

def test_case_0():
    var_0 = {}
    var_1 = 'MockDestinationCls'
    var_2 = {}
    var_3 = 'nonexistent'
    var_4 = 5

def test_case_0():
    var_0 = lambda x: x * 2
    var_1 = lambda self, value: (True, None)
    var_2 = 'key'
    var_3 = {}
    var_4 = set()
    var_5 = 'key'
    var_6 = 5



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_precord_repr_returns_correct_format. Retrieved 8/14 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'value1'
    var_8 = 123
    var_9 = 'TestRecord('
    var_10 = ')'
    var_11 = ','



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Failed to parse test_persistent_returns_instance_of_destination_class.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_failure. Retrieved 2/9 statements.
# Failed to parse test_persistent_raises_invariant_exception_on_missing_mandatory_fields.
# Partially parsed test_persistent_calls_check_global_invariants. Retrieved 2/11 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_type. Retrieved 1/6 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 3/9 statements.
# Partially parsed test_persistent_aggregates_multiple_invariant_errors. Retrieved 4/13 statements.
# Partially parsed test_persistent_aggregates_missing_fields_and_invariant_errors. Retrieved 2/10 statements.


def test_case_0():
    var_0 = 'field'
    var_1 = -1

def test_case_0():
    var_0 = 'field'
    var_1 = -1

def test_case_0():
    var_0 = 1

def test_case_0():
    var_0 = 1
    var_1 = 'field'
    var_2 = 2

def test_case_0():
    var_0 = 'field1'
    var_1 = -1
    var_2 = 'field2'
    var_3 = 1

def test_case_0():
    var_0 = 'field2'
    var_1 = -1



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_set_with_valid_field_and_factory. Retrieved 3/16 statements.
# Partially parsed test_set_with_valid_field_and_no_factory. Retrieved 4/17 statements.
# Partially parsed test_set_with_invalid_field. Retrieved 4/11 statements.
# Partially parsed test_set_with_factory_invariant_exception. Retrieved 3/19 statements.
# Partially parsed test_set_with_field_invariant_failure. Retrieved 3/17 statements.
# Partially parsed test_set_with_type_check_failure. Retrieved 3/16 statements.
# Partially parsed test_set_with_ignore_extra_complaint. Retrieved 6/23 statements.


def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = set()
    var_3 = 'key'
    var_4 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'invalid_key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = 'key'
    var_1 = {}
    var_2 = 'key'
    var_3 = 'not_an_int'

def test_case_0():
    var_0 = None
    var_1 = lambda x, ignore_extra=False: var_0
    var_2 = 'key'
    var_3 = {}
    var_4 = True
    var_5 = 'key'
    var_6 = 5



# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 8/12 statements.
# Partially parsed test_serialize_without_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_serialize_with_format_argument. Retrieved 6/10 statements.
# Partially parsed test_serialize_with_none_format. Retrieved 4/8 statements.
# Partially parsed test_serialize_empty_record. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_multiple_fields_mixed_serializers. Retrieved 11/16 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'alice'
    var_2 = 30
    var_3 = module_0.serialize()
    var_4 = 'name'
    var_5 = 'age'
    var_6 = 'ALICE'
    var_7 = {var_4: var_6, var_5: var_2}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'alice'
    var_3 = 30
    var_4 = module_0.serialize()
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {var_5: var_2, var_6: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'fmt'
    var_2 = module_0.serialize(var_1)
    var_3 = 'value'
    var_4 = 'fmt:42'
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.serialize()
    var_2 = 'value'
    var_3 = {var_2: var_0}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.serialize()
    var_1 = {}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 10
    var_2 = 'test'
    var_3 = 3.14
    var_4 = module_0.serialize()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = 20
    var_9 = '3.14'
    var_10 = {var_5: var_8, var_6: var_2, var_7: var_9}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test___new___sets_fields_and_invariants. Retrieved 13/38 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'base1_field'
    var_2 = lambda self: (True, ())
    var_3 = 'base2_field'
    var_4 = lambda self: (False, ('error',))
    var_5 = 'custom_field'
    var_6 = False
    var_7 = 'default'
    var_8 = 'TestClass'
    var_9 = '_precord_fields'
    var_10 = '_precord_invariants'
    var_11 = None
    var_12 = 1
    var_13 = '_precord_mandatory_fields'
    var_14 = '_precord_initial_values'



# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------

# Partially parsed test___new___sets_fields_and_invariants. Retrieved 9/19 statements.
# Partially parsed test___new___inherits_fields_and_invariants. Retrieved 6/18 statements.
# Partially parsed test___new___handles_no_fields. Retrieved 4/5 statements.
# Partially parsed test___new___wraps_invariants. Retrieved 6/21 statements.
# Partially parsed test___new___raises_on_non_callable_invariant. Retrieved 3/7 statements.


def test_case_0():
    var_0 = True
    var_1 = 10
    var_2 = False
    var_3 = None
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = ()
    var_7 = 'TestClass'
    var_8 = '_precord_invariants'

def test_case_0():
    var_0 = 'base_field'
    var_1 = lambda self: (True, ())
    var_2 = 'new_field'
    var_3 = True
    var_4 = 20
    var_5 = 'DerivedClass'
    var_6 = '_precord_invariants'

def test_case_0():
    var_0 = {}
    var_1 = ()
    var_2 = 'EmptyClass'
    var_3 = set()

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'TestClass'
    var_2 = '_precord_invariants'
    var_3 = 0
    var_4 = 1
    var_5 = None

def test_case_0():
    var_0 = 'not callable'
    var_1 = {}
    var_2 = 'TestClass'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_persistent_returns_instance_of_destination_class. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_missing_mandatory_fields. Retrieved 4/12 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_field_invariant_errors. Retrieved 5/15 statements.
# Partially parsed test_persistent_raises_invariant_exception_on_global_invariant_failure. Retrieved 4/12 statements.
# Partially parsed test_persistent_returns_same_instance_if_not_dirty_and_already_correct_class. Retrieved 3/11 statements.
# Partially parsed test_persistent_creates_new_instance_if_dirty. Retrieved 5/15 statements.


def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []
    var_3 = {}

def test_case_0():
    var_0 = {}
    var_1 = 'mandatory_field'
    var_2 = {var_1}
    var_3 = []
    var_4 = {}

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = lambda x: (False, 'global_error')
    var_3 = [var_2]
    var_4 = {}

def test_case_0():
    var_0 = {}
    var_1 = set()
    var_2 = []

def test_case_0():
    var_0 = 'field'
    var_1 = set()
    var_2 = []
    var_3 = {}
    var_4 = 'field'
    var_5 = 'value'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_set_with_valid_field_and_value. Retrieved 4/23 statements.
# Partially parsed test_set_with_field_factory_ignore_extra. Retrieved 5/24 statements.
# Partially parsed test_set_invokes_check_type. Retrieved 4/24 statements.
# Partially parsed test_set_invariant_fails. Retrieved 4/23 statements.
# Partially parsed test_set_factory_raises_invariant_exception. Retrieved 4/23 statements.
# Partially parsed test_set_with_non_existent_field. Retrieved 4/12 statements.
# Partially parsed test_set_with_factory_fields_skips_factory. Retrieved 7/27 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = True
    var_3 = 'key'
    var_4 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'key'
    var_3 = 'not_an_int'

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'key'
    var_3 = 5

def test_case_0():
    var_0 = {}
    var_1 = {}
    var_2 = 'nonexistent'
    var_3 = 5

def test_case_0():
    var_0 = {}
    var_1 = []
    var_2 = {}
    var_3 = set()
    var_4 = 'key'
    var_5 = 5
    var_6 = len(var_1)
    assert var_6 == 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test___new___creates_record_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test___new___creates_record_with_initial_values. Retrieved 1/4 statements.
# Partially parsed test___new___creates_record_with_overridden_initial_values. Retrieved 2/5 statements.
# Partially parsed test___new___creates_record_with_callable_initial_value. Retrieved 1/4 statements.
# Partially parsed test___new___creates_record_with_factory_fields. Retrieved 3/8 statements.
# Partially parsed test___new___creates_record_with_ignore_extra. Retrieved 2/5 statements.
# Partially parsed test___new___raises_attribute_error_for_unknown_field. Retrieved 1/5 statements.
# Partially parsed test___new___raises_invariant_exception_for_invalid_field. Retrieved 1/5 statements.
# Partially parsed test___new___raises_invariant_exception_for_missing_mandatory_field. Retrieved 1/5 statements.
# Partially parsed test___new___creates_record_with_global_invariant. Retrieved 3/7 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field1'
    var_3 = 'field2'
    var_4 = 10
    var_5 = 'default'
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 20

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field1'
    var_2 = lambda : 42
    var_3 = {var_1: var_2}

def test_case_0():
    var_0 = 'field1'
    var_1 = 2
    var_2 = lambda x: x * var_1
    var_3 = 5

def test_case_0():
    var_0 = 'field1'
    var_1 = True
    var_2 = 2

def test_case_0():
    var_0 = 'field1'
    var_1 = 2

def test_case_0():
    var_0 = 'field1'
    var_1 = -1

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field1'
    var_2 = {var_1}

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = lambda r: (r['field1'] + r['field2'] == 10, 'SUM_ERR')
    var_3 = [var_2]
    var_4 = 3
    var_5 = 8



# Parsed testcases at query #4
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 3/18 statements.
# Partially parsed test___new___moves_pfield_instances_to_fields. Retrieved 4/12 statements.
# Partially parsed test___new___stores_invariants_correctly. Retrieved 5/20 statements.
# Partially parsed test___new___wraps_invariants_that_return_multiple_results. Retrieved 5/11 statements.
# Partially parsed test___new___sets_mandatory_fields. Retrieved 7/22 statements.
# Partially parsed test___new___sets_initial_values. Retrieved 6/21 statements.
# Partially parsed test___new___sets_empty_slots. Retrieved 2/3 statements.
# Partially parsed test___new___inherits_fields_and_invariants. Retrieved 7/30 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'base1_field'
    var_2 = 'base2_field'
    var_3 = 'custom_field'
    var_4 = '_precord_fields'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'pfield_key'
    var_2 = ()
    var_3 = '_precord_fields'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = '_precord_invariants'
    var_2 = 0
    var_3 = None
    var_4 = 1

def test_case_0():
    var_0 = '__invariant__'
    var_1 = ()
    var_2 = '_precord_invariants'
    var_3 = 0
    var_4 = None

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not a callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = '_precord_invariants'
    var_5 = '__invariant__'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = True
    var_2 = False
    var_3 = 'mandatory'
    var_4 = 'non_mandatory'
    var_5 = ()
    var_6 = '_precord_fields'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'default'
    var_2 = 'with_initial'
    var_3 = 'without_initial'
    var_4 = ()
    var_5 = '_precord_fields'

def test_case_0():
    var_0 = {}
    var_1 = ()

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'grand_field'
    var_2 = 'parent_field'
    var_3 = 'child_field'
    var_4 = '_precord_fields'
    var_5 = '_precord_invariants'
    var_6 = '__invariant__'
    var_7 = 0
    var_8 = None



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 4/6 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 4/6 statements.
# Partially parsed test_precord_new_without_ignore_extra_raises. Retrieved 3/6 statements.
# Failed to parse test_precord_new_with_initial_values_from_class.
# Partially parsed test_precord_new_overrides_initial_values. Retrieved 1/4 statements.
# Failed to parse test_precord_new_with_mandatory_fields_missing.
# Partially parsed test_precord_new_with_invariant_failure. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_global_invariant_failure. Retrieved 4/9 statements.
# Partially parsed test_precord_new_with_factory_and_invariant. Retrieved 1/4 statements.
# Partially parsed test_precord_new_with_factory_exception. Retrieved 1/5 statements.
# Partially parsed test_precord_new_returns_same_instance_if_no_changes. Retrieved 2/7 statements.
# Partially parsed test_precord_new_with_check_type_violation. Retrieved 1/5 statements.
# Partially parsed test_precord_new_with_ignore_extra_and_factory. Retrieved 3/6 statements.
# Partially parsed test_precord_new_with_factory_fields_list. Retrieved 4/9 statements.
# Failed to parse test_precord_new_with_empty_initial.
# Failed to parse test_precord_new_with_callable_initial.
# Partially parsed test_precord_new_with_multiple_fields_set. Retrieved 6/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 1
    var_3 = 2

def test_case_0():
    var_0 = '5'
    var_1 = None

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2

def test_case_0():
    var_0 = 30

def test_case_0():
    var_0 = -5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 10
    var_3 = 5

def test_case_0():
    var_0 = '5'

def test_case_0():
    var_0 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1

def test_case_0():
    var_0 = 'string'

def test_case_0():
    var_0 = '10'
    var_1 = 20
    var_2 = True

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = '5'
    var_2 = 10
    var_3 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = module_0.field()
    var_3 = 1
    var_4 = 2
    var_5 = 3



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_new_without_precord_size_and_buckets. Retrieved 2/5 statements.
# Partially parsed test_new_with_factory_fields_and_ignore_extra. Retrieved 5/8 statements.
# Partially parsed test_new_with_initial_values. Retrieved 2/5 statements.
# Partially parsed test_new_with_kwargs_overrides_initial_values. Retrieved 3/6 statements.
# Partially parsed test_new_with_no_special_attributes. Retrieved 2/4 statements.
# Partially parsed test_new_with_regular_kwargs_only. Retrieved 2/5 statements.
# Partially parsed test_new_with_empty_kwargs. Retrieved 2/4 statements.
# Partially parsed test_new_with_multiple_fields. Retrieved 3/6 statements.


def test_case_0():
    var_0 = {}
    var_1 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = 1
    var_3 = 'a'
    var_4 = {var_3}
    var_5 = True

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'b'
    var_3 = lambda : 2
    var_4 = {var_2: var_3}
    var_5 = 1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'b'
    var_3 = lambda : 2
    var_4 = {var_2: var_3}
    var_5 = 1
    var_6 = 3

def test_case_0():
    var_0 = {}
    var_1 = {}

def test_case_0():
    var_0 = 'x'
    var_1 = {}
    var_2 = 10

def test_case_0():
    var_0 = {}
    var_1 = {}

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = {}
    var_3 = 'test'
    var_4 = 42



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_precord_constructor_without_special_attributes. Retrieved 5/8 statements.
# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 13/16 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_precord_constructor_initial_values_overridden. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/9 statements.
# Partially parsed test_precord_constructor_ignore_extra. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_ignore_extra_false. Retrieved 7/10 statements.
# Partially parsed test_precord_constructor_empty_record. Retrieved 3/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 4/11 statements.
# Partially parsed test_precord_constructor_initial_value_not_callable. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 1
    var_8 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 2
    var_8 = 'x'
    var_9 = hash(var_8)
    var_10 = 1
    var_11 = (var_9, var_8, var_10)
    var_12 = 'y'
    var_13 = hash(var_12)
    var_14 = 2
    var_15 = (var_13, var_12, var_14)
    var_16 = [var_11, var_15]

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'x'
    var_7 = 'y'
    var_8 = lambda : 10
    var_9 = 20
    var_10 = {var_6: var_8, var_7: var_9}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'x'
    var_7 = 'y'
    var_8 = lambda : 10
    var_9 = 20
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'x'
    var_8 = '5'
    var_9 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = True
    var_8 = 2
    var_9 = 3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = False
    var_8 = 1
    var_9 = 2
    var_10 = 3

def test_case_0():
    var_0 = ()
    var_1 = {}
    var_2 = {}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 0
    var_1 = [var_0]
    var_2 = ()
    var_3 = 'x'
    var_4 = module_0.field()
    var_5 = {var_3: var_4}
    var_6 = 'x'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = 42
    var_6 = {var_4: var_5}



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_repr_with_no_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_integer_field_names. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 'test'
    var_4 = "TestRecord(x=10, y='test')"

def test_case_0():
    var_0 = {}
    var_1 = 'TestRecord()'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = "TestRecord(data={'key': 'value'})"

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '1'
    var_3 = '2'
    var_4 = 100
    var_5 = 200
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'TestRecord(1=100, 2=200)'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test___new___creates_precord_with_special_attributes. Retrieved 2/9 statements.
# Partially parsed test___new___uses_evolver_for_initial_values. Retrieved 3/6 statements.
# Partially parsed test___new___applies_initial_values_from_class. Retrieved 2/5 statements.
# Partially parsed test___new___overrides_initial_values_with_kwargs. Retrieved 3/6 statements.
# Partially parsed test___new___handles_factory_fields_parameter. Retrieved 4/7 statements.
# Partially parsed test___new___handles_ignore_extra_parameter. Retrieved 4/7 statements.
# Partially parsed test___new___raises_attribute_error_for_unknown_field. Retrieved 2/6 statements.
# Partially parsed test___new___raises_invariant_exception_for_invalid_value. Retrieved 2/6 statements.
# Partially parsed test___new___raises_invariant_exception_for_missing_mandatory_field. Retrieved 2/6 statements.
# Partially parsed test___new___creates_empty_precord_without_initial_values. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 10
    var_4 = 'test'

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = lambda : 5
    var_4 = {var_2: var_3}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = lambda : 5
    var_4 = {var_2: var_3}
    var_5 = 10

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 5
    var_3 = 'field1'
    var_4 = {var_3}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 5
    var_3 = 20
    var_4 = True

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 10

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = -5

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = {var_2}

def test_case_0():
    var_0 = ()
    var_1 = {}



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_precord_repr_with_single_field. Retrieved 2/6 statements.
# Partially parsed test_precord_repr_with_multiple_fields. Retrieved 3/7 statements.
# Partially parsed test_precord_repr_with_no_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_repr_with_nested_values. Retrieved 4/8 statements.
# Partially parsed test_precord_repr_with_integer_field_names. Retrieved 6/10 statements.


def test_case_0():
    var_0 = 'name'
    var_1 = 'Alice'
    var_2 = "TestRecord(name='Alice')"

def test_case_0():
    var_0 = 'x'
    var_1 = 'y'
    var_2 = 10
    var_3 = 'test'
    var_4 = "TestRecord(x=10, y='test')"

def test_case_0():
    var_0 = {}
    var_1 = 'TestRecord()'

def test_case_0():
    var_0 = 'data'
    var_1 = 'key'
    var_2 = 'value'
    var_3 = {var_1: var_2}
    var_4 = "TestRecord(data={'key': 'value'})"

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '1'
    var_3 = '2'
    var_4 = 100
    var_5 = 200
    var_6 = {var_2: var_4, var_3: var_5}
    var_7 = 'TestRecord(1=100, 2=200)'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_precord_constructor_with_special_attributes. Retrieved 8/11 statements.
# Partially parsed test_precord_constructor_with_initial_values. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_callable_initial_value. Retrieved 3/5 statements.
# Partially parsed test_precord_constructor_with_factory_fields. Retrieved 6/8 statements.
# Partially parsed test_precord_constructor_with_ignore_extra. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_without_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_constructor_with_multiple_fields. Retrieved 5/7 statements.
# Partially parsed test_precord_constructor_with_no_fields. Retrieved 2/5 statements.
# Partially parsed test_precord_constructor_with_overriding_initial_values. Retrieved 4/6 statements.
# Partially parsed test_precord_constructor_with_empty_kwargs. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 1
    var_5 = 'x'
    var_6 = 10
    var_7 = (var_5, var_6)
    var_8 = (var_7,)
    var_9 = (var_8,)

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'y'
    var_7 = 20
    var_8 = {var_6: var_7}
    var_9 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = lambda : 100
    var_6 = {var_4: var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = 5
    var_6 = {var_4: var_5}
    var_7 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = True
    var_5 = 10
    var_6 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 10
    var_5 = 20

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 'c'
    var_4 = module_0.field()
    var_5 = module_0.field()
    var_6 = module_0.field()
    var_7 = {var_1: var_4, var_2: var_5, var_3: var_6}
    var_8 = 1
    var_9 = 2
    var_10 = 3

def test_case_0():
    var_0 = ()
    var_1 = {}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = 'y'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = 'x'
    var_7 = 'y'
    var_8 = 1
    var_9 = 2
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = 10

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'x'
    var_2 = module_0.field()
    var_3 = {var_1: var_2}
    var_4 = 'x'
    var_5 = 5
    var_6 = {var_4: var_5}



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_precord_repr_returns_correct_format. Retrieved 10/19 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = module_0.field()
    var_4 = module_0.field()
    var_5 = {var_1: var_3, var_2: var_4}
    var_6 = {}
    var_7 = 'value1'
    var_8 = 123
    var_9 = 'TestRecord('
    var_10 = ')'
    var_11 = len(var_9)
    var_12 = len(var_10)
    var_13 = ', '



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test___new___creates_fields_and_invariants. Retrieved 9/28 statements.
# Partially parsed test___new___handles_no_invariants. Retrieved 4/16 statements.
# Partially parsed test___new___wraps_invariants. Retrieved 7/31 statements.
# Partially parsed test___new___raises_on_non_callable_invariant. Retrieved 4/17 statements.
# Partially parsed test___new___handles_mandatory_and_initial_fields. Retrieved 8/22 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = lambda self: (True, ())
    var_2 = 'base_field'
    var_3 = lambda self: (False, ('error',))
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = True
    var_7 = 'default'
    var_8 = 'TestClass'
    var_9 = '_precord_invariants'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = {}
    var_3 = 'TestClass'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = '__invariant__'
    var_2 = 'TestClass'
    var_3 = '_precord_invariants'
    var_4 = 0
    var_5 = 'dummy'
    var_6 = 1

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'not callable'
    var_2 = {}
    var_3 = 'TestClass'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'mandatory_field'
    var_2 = 'initial_field'
    var_3 = 'regular_field'
    var_4 = True
    var_5 = 42
    var_6 = ()
    var_7 = 'TestClass'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 3/13 statements.
# Partially parsed test___new___handles_mandatory_fields. Retrieved 7/19 statements.
# Partially parsed test___new___handles_initial_values. Retrieved 6/18 statements.
# Partially parsed test___new___stores_invariants. Retrieved 5/20 statements.
# Partially parsed test___new___raises_on_non_callable_invariant. Retrieved 5/10 statements.
# Partially parsed test___new___sets_slots. Retrieved 5/10 statements.
# Partially parsed test___new___inherits_fields_and_invariants. Retrieved 5/23 statements.


import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'base_field'
    var_2 = 'new_field'
    var_3 = '_precord_fields'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'mandatory_field'
    var_2 = 'optional_field'
    var_3 = True
    var_4 = False
    var_5 = ()
    var_6 = '_precord_fields'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'with_initial'
    var_2 = 'without_initial'
    var_3 = 'default'
    var_4 = ()
    var_5 = '_precord_fields'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = '_precord_invariants'
    var_2 = 0
    var_3 = None
    var_4 = 1

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not a callable'
    var_2 = {var_0: var_1}
    var_3 = '_precord_invariants'
    var_4 = '__invariant__'

import builtins as module_0
import pyrsistent._field_common as module_1

def test_case_0():
    var_0 = module_0.object()
    var_1 = {}
    var_2 = ()
    var_3 = '_precord_fields'
    var_4 = module_1.set_fields(var_1, var_2, var_3)

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = 'grand_field'
    var_2 = 'base_field'
    var_3 = 'child_field'
    var_4 = '_precord_fields'
    var_5 = '_precord_invariants'
    var_6 = '__invariant__'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_precord_new_creates_instance_with_special_attributes. Retrieved 2/8 statements.
# Failed to parse test_precord_new_uses_evolver_for_initial_values.
# Partially parsed test_precord_new_applies_initial_values. Retrieved 4/9 statements.
# Partially parsed test_precord_new_overrides_initial_values_with_kwargs. Retrieved 5/10 statements.
# Partially parsed test_precord_new_passes_factory_fields_to_evolver. Retrieved 1/7 statements.
# Partially parsed test_precord_new_passes_ignore_extra_to_evolver. Retrieved 1/7 statements.
# Partially parsed test_precord_new_handles_callable_initial_values. Retrieved 4/9 statements.
# Partially parsed test_precord_new_handles_non_callable_initial_values. Retrieved 3/8 statements.


def test_case_0():
    var_0 = 0
    var_1 = []

import builtins as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.object()
    var_2 = 'default'
    var_3 = lambda : var_2

import builtins as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.object()
    var_2 = 'default'
    var_3 = lambda : var_2
    var_4 = 'custom'

def test_case_0():
    var_0 = set()

def test_case_0():
    var_0 = True

import builtins as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.object()
    var_2 = 'callable_result'
    var_3 = lambda : var_2

import builtins as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = module_0.object()
    var_2 = 'static_value'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_precord_new_with_special_attributes. Retrieved 5/12 statements.
# Partially parsed test_precord_new_with_initial_values. Retrieved 3/5 statements.
# Partially parsed test_precord_new_with_default_initial_values. Retrieved 2/4 statements.
# Partially parsed test_precord_new_with_overridden_defaults. Retrieved 3/5 statements.
# Partially parsed test_precord_new_with_factory_fields. Retrieved 3/8 statements.
# Partially parsed test_precord_new_with_ignore_extra. Retrieved 3/5 statements.
# Partially parsed test_precord_new_with_invalid_field_raises_attribute_error. Retrieved 2/5 statements.
# Partially parsed test_precord_new_with_field_invariant_failure. Retrieved 1/7 statements.
# Partially parsed test_precord_new_with_mandatory_fields. Retrieved 2/6 statements.
# Partially parsed test_precord_new_with_global_invariant_failure. Retrieved 3/9 statements.


def test_case_0():
    var_0 = 0
    var_1 = None
    var_2 = [var_1]
    var_3 = 8
    var_4 = var_2 * var_3

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 1
    var_6 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = lambda : 10
    var_8 = 20
    var_9 = {var_5: var_7, var_6: var_8}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = lambda : 10
    var_8 = 20
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 100

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'a'
    var_3 = 5
    var_4 = 'hello'

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = True
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 2

def test_case_0():
    var_0 = 'a'
    var_1 = -1

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'a'
    var_3 = {var_2}
    var_4 = 2

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = -1
    var_6 = -1



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_serialize_without_custom_serializer. Retrieved 7/9 statements.
# Partially parsed test_serialize_with_custom_serializer. Retrieved 7/12 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 8/13 statements.
# Partially parsed test_serialize_with_multiple_fields_and_serializers. Retrieved 10/17 statements.
# Partially parsed test_serialize_empty_record. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = None
    var_3 = None
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'value1'
    var_6 = 123
    var_7 = module_0.serialize()
    var_8 = 'field1'
    var_9 = 'field2'
    var_10 = {var_8: var_5, var_9: var_6}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = module_0.serialize()
    var_5 = 'field1'
    var_6 = 'field2'
    var_7 = 'custom_value1'
    var_8 = {var_5: var_7, var_6: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'value1'
    var_3 = 123
    var_4 = 'json'
    var_5 = module_0.serialize(var_4)
    var_6 = 'field1'
    var_7 = 'field2'
    var_8 = 'json_value1'
    var_9 = {var_6: var_8, var_7: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'field1'
    var_1 = 'field2'
    var_2 = 'field3'
    var_3 = 5
    var_4 = 20
    var_5 = 'test'
    var_6 = module_0.serialize()
    var_7 = 'field1'
    var_8 = 'field2'
    var_9 = 'field3'
    var_10 = 10
    var_11 = 30
    var_12 = {var_7: var_10, var_8: var_11, var_9: var_5}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.serialize()
    var_2 = {}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_precord_initial_values_used_when_present. Retrieved 2/4 statements.
# Partially parsed test_precord_initial_values_ignored_when_overridden. Retrieved 3/5 statements.
# Partially parsed test_precord_initial_values_with_callable. Retrieved 2/4 statements.
# Partially parsed test_precord_initial_values_with_non_callable. Retrieved 2/4 statements.
# Partially parsed test_precord_initial_values_empty_dict_no_effect. Retrieved 3/5 statements.
# Partially parsed test_precord_initial_values_combined_with_kwargs. Retrieved 3/5 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = lambda : 42
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = lambda : 42
    var_5 = {var_3: var_4}
    var_6 = 100

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = lambda : 'default'
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = 'a'
    var_4 = 'static_default'
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = module_0.field()
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = 5

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = module_0.field()
    var_3 = module_0.field()
    var_4 = {var_0: var_2, var_1: var_3}
    var_5 = 'a'
    var_6 = 'b'
    var_7 = lambda : 1
    var_8 = lambda : 2
    var_9 = {var_5: var_7, var_6: var_8}
    var_10 = 20



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_precord_new_creates_instance_with_special_attributes. Retrieved 5/11 statements.
# Partially parsed test_precord_new_uses_evolver_for_initial_values. Retrieved 4/7 statements.
# Partially parsed test_precord_new_applies_initial_values_from_class. Retrieved 2/5 statements.
# Partially parsed test_precord_new_overrides_initial_values_with_kwargs. Retrieved 3/6 statements.
# Partially parsed test_precord_new_handles_factory_fields_parameter. Retrieved 3/6 statements.
# Partially parsed test_precord_new_handles_ignore_extra_parameter. Retrieved 4/7 statements.
# Partially parsed test_precord_new_raises_attribute_error_for_unknown_field. Retrieved 2/6 statements.
# Partially parsed test_precord_new_invokes_field_factory_with_ignore_extra. Retrieved 4/7 statements.
# Partially parsed test_precord_new_raises_invariant_exception_for_missing_mandatory_fields. Retrieved 2/6 statements.
# Partially parsed test_precord_new_raises_invariant_exception_for_field_invariant_failure. Retrieved 2/8 statements.
# Partially parsed test_precord_new_raises_invariant_exception_for_global_invariant_failure. Retrieved 3/10 statements.


def test_case_0():
    var_0 = None
    var_1 = [var_0]
    var_2 = 8
    var_3 = var_1 * var_2
    var_4 = 0

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = {}
    var_4 = 42
    var_5 = 'test'

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = lambda : 100
    var_6 = 'default'
    var_7 = {var_3: var_5, var_4: var_6}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = 'field1'
    var_4 = 'field2'
    var_5 = lambda : 100
    var_6 = 'default'
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = 200

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = None
    var_3 = 21

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = True
    var_3 = 10
    var_4 = 20

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 10

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = True
    var_3 = 'a'
    var_4 = {var_3: var_2}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field1'
    var_3 = {var_2}

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = -5

def test_case_0():
    var_0 = ()
    var_1 = 'field1'
    var_2 = 'field2'
    var_3 = -10
    var_4 = 5



# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------

# Partially parsed test___new___sets_fields_correctly. Retrieved 8/16 statements.
# Partially parsed test___new___inherits_fields_from_bases. Retrieved 3/6 statements.
# Partially parsed test___new___merges_fields_from_multiple_bases. Retrieved 4/8 statements.
# Partially parsed test___new___stores_invariants. Retrieved 5/20 statements.
# Partially parsed test___new___wraps_invariants_correctly. Retrieved 5/11 statements.
# Partially parsed test___new___sets_mandatory_fields. Retrieved 7/19 statements.
# Partially parsed test___new___sets_initial_values. Retrieved 7/19 statements.
# Partially parsed test___new___sets_empty_slots. Retrieved 2/3 statements.


def test_case_0():
    var_0 = True
    var_1 = 10
    var_2 = False
    var_3 = None
    var_4 = 'field1'
    var_5 = 'field2'
    var_6 = ()
    var_7 = '_precord_fields'

def test_case_0():
    var_0 = 'base_field'
    var_1 = 'base_value'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = '_precord_fields'

def test_case_0():
    var_0 = 'field1'
    var_1 = 'value1'
    var_2 = {var_0: var_1}
    var_3 = 'field2'
    var_4 = 'value2'
    var_5 = {var_3: var_4}
    var_6 = {}
    var_7 = '_precord_fields'

def test_case_0():
    var_0 = '__invariant__'
    var_1 = '_precord_invariants'
    var_2 = 0
    var_3 = None
    var_4 = 1

def test_case_0():
    var_0 = '__invariant__'
    var_1 = ()
    var_2 = '_precord_invariants'
    var_3 = 0
    var_4 = None

def test_case_0():
    var_0 = True
    var_1 = None
    var_2 = False
    var_3 = 'field_mandatory'
    var_4 = 'field_optional'
    var_5 = ()
    var_6 = '_precord_fields'

import builtins as module_0

def test_case_0():
    var_0 = module_0.object()
    var_1 = False
    var_2 = 5
    var_3 = 'field_with_initial'
    var_4 = 'field_no_initial'
    var_5 = ()
    var_6 = '_precord_fields'

def test_case_0():
    var_0 = {}
    var_1 = ()

import pyrsistent._checked_types as module_0

def test_case_0():
    var_0 = '__invariant__'
    var_1 = 'not a callable'
    var_2 = {var_0: var_1}
    var_3 = ()
    var_4 = '_precord_invariants'
    var_5 = '__invariant__'
    var_6 = module_0.store_invariants(var_2, var_3, var_4, var_5)



# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------

# Partially parsed test_serialize_with_custom_serializer. Retrieved 8/13 statements.
# Partially parsed test_serialize_without_custom_serializer. Retrieved 8/11 statements.
# Partially parsed test_serialize_with_format_parameter. Retrieved 6/11 statements.
# Partially parsed test_serialize_with_none_format. Retrieved 4/9 statements.
# Partially parsed test_serialize_on_empty_record. Retrieved 2/6 statements.
# Partially parsed test_serialize_with_multiple_fields_mixed_serializers. Retrieved 11/17 statements.


import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 'test'
    var_2 = 42
    var_3 = module_0.serialize()
    var_4 = 'name'
    var_5 = 'value'
    var_6 = 'serialized_42'
    var_7 = {var_4: var_1, var_5: var_6}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = module_0.field()
    var_2 = 'Alice'
    var_3 = 30
    var_4 = module_0.serialize()
    var_5 = 'name'
    var_6 = 'age'
    var_7 = {var_5: var_2, var_6: var_3}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'example'
    var_1 = 'json'
    var_2 = module_0.serialize(var_1)
    var_3 = 'data'
    var_4 = 'json:example'
    var_5 = {var_3: var_4}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.serialize()
    var_2 = 'item'
    var_3 = {var_2: var_0}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.serialize()
    var_1 = {}

import pyrsistent._field_common as module_0

def test_case_0():
    var_0 = module_0.field()
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = module_0.serialize()
    var_5 = 'a'
    var_6 = 'b'
    var_7 = 'c'
    var_8 = 4
    var_9 = '3'
    var_10 = {var_5: var_1, var_6: var_8, var_7: var_9}



# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------






