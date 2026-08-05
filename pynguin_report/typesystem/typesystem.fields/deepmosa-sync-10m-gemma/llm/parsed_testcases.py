####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  hello  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  hello  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hel\x00lo'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcd'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(allow_blank=var_1, coerce_types=var_0, **var_3)
    var_5 = '  '
    var_6 = var_4.validate(var_5)
    assert var_6 is None



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Field'
    var_5 = 'title'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = var_7.choices
    var_9 = bool(var_7.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')])
    assert var_9 is True
    var_10 = var_7.title
    assert var_10 == 'Test Field'
    var_11 = var_7.coerce_types
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'val1'
    var_1 = 'Label 1'
    var_2 = (var_0, var_1)
    var_3 = 'val2'
    var_4 = 'Label 2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'Tuple Field'
    var_8 = 'title'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.choices
    var_12 = bool(var_10.choices == [('val1', 'Label 1'), ('val2', 'Label 2')])
    assert var_12 is True
    var_13 = var_10.title
    assert var_13 == 'Tuple Field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = 'Empty Field'
    var_2 = 'title'
    var_3 = {var_2: var_1}
    var_4 = module_0.Choice(choices=var_0, **var_3)
    var_5 = var_4.choices
    var_6 = bool(var_4.choices == [])
    assert var_6 is True
    var_7 = var_4.title
    assert var_7 == 'Empty Field'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'Desc'
    var_3 = True
    var_4 = 'description'
    var_5 = 'allow_null'
    var_6 = 'read_only'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Choice(choices=var_1, **var_7)
    var_9 = var_8.description
    assert var_9 == 'Desc'
    var_10 = var_8.allow_null
    assert var_10 is True
    var_11 = var_8.read_only
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Choice(choices=var_1, coerce_types=var_2, **var_3)
    var_5 = var_4.coerce_types
    assert var_5 is False

def test_case_0():
    pass



# Parsed testcases at query #3
#--------------------------




import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'Test'
    var_4 = 10
    var_5 = module_1.Field(title=var_3, default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 == 10

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'Test'
    var_4 = 'dynamic'
    var_5 = lambda : var_4
    var_6 = module_1.Field(title=var_3, default=var_5)
    var_7 = var_6.get_default_value()
    assert var_7 == 'dynamic'

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'Test'
    var_4 = module_1.Field(title=var_3, default=var_2)
    var_5 = var_4.get_default_value()
    assert var_5 is None

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = []
    var_1 = {}
    var_2 = module_0.object(*var_0, **var_1)
    var_3 = 'Test'
    var_4 = None
    var_5 = module_1.Field(title=var_3, default=var_4)
    var_6 = var_5.get_default_value()
    assert var_6 is None



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_validate_string_conversion. Retrieved 4/5 statements.
# Partially parsed test_validate_precision_rounding. Retrieved 5/6 statements.
# Partially parsed test_validate_numeric_type_int_enforcement_pass. Retrieved 1/3 statements.
# Partially parsed test_validate_numeric_type_int_enforcement_fail. Retrieved 1/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 10.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123.45'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    var_5 = 'minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5.1
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 5.1)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5
    var_4 = var_2.validate(var_3)
    var_5 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = 11
    var_4 = var_2.validate(var_3)
    var_5 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9.9
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 9.9)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    var_5 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    assert var_4 == 4

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 3
    var_4 = var_2.validate(var_3)
    var_5 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 1.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 0.7
    var_4 = var_2.validate(var_3)
    var_5 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = 1.2345
    var_4 = var_2.validate(var_3)
    var_5 = '1.23'

def test_case_0():
    var_0 = 10.0

def test_case_0():
    var_0 = 10.5
    var_1 = 'integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = 'finite'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '10'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'



# Parsed testcases at query #5
#--------------------------




import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = ()
    var_2 = 'allow_null'
    var_3 = 'validate_or_error'
    var_4 = True
    var_5 = None
    var_6 = (var_5, var_5)
    var_7 = lambda self, v: var_6
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Union(var_13, **var_14)
    var_16 = var_15.validate(var_5)
    assert var_16 is None

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = ()
    var_2 = 'allow_null'
    var_3 = 'validate_or_error'
    var_4 = False
    var_5 = None
    var_6 = (var_5, var_5)
    var_7 = lambda self, v: var_6
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Union(var_13, **var_14)
    var_16 = None
    var_17 = var_15.validate(var_16)
    var_18 = 'null'
    var_19 = bool('null' in str(e).lower())
    assert var_19 is True

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Child'
    var_1 = ()
    var_2 = 'validate_or_error'
    var_3 = None
    var_4 = 'type error'
    var_5 = [var_4]
    var_6 = {}
    var_7 = module_0.Exception(*var_5, **var_6)
    var_8 = (var_3, var_7)
    var_9 = lambda self, v: var_8
    var_10 = {var_2: var_9}
    var_11 = [var_0, var_1, var_10]
    var_12 = {}
    var_13 = module_0.type(*var_11, **var_12)
    var_14 = var_13()
    var_15 = ()
    var_16 = 'success'
    var_17 = (var_16, var_3)
    var_18 = lambda self, v: var_17
    var_19 = {var_2: var_18}
    var_20 = [var_0, var_15, var_19]
    var_21 = {}
    var_22 = module_0.type(*var_20, **var_21)
    var_23 = var_22()
    var_24 = [var_14, var_23]
    var_25 = {}
    var_26 = module_1.Union(var_24, **var_25)
    var_27 = 'some_value'
    var_28 = var_26.validate(var_27)
    assert var_28 == 'success'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_array_serialize_with_single_field_type_int. Retrieved 4/10 statements.
# Partially parsed test_array_serialize_with_list_of_fields. Retrieved 11/26 statements.
# Partially parsed test_array_serialize_with_list_of_fields_correct_mapping. Retrieved 3/14 statements.
# Partially parsed test_array_serialize_with_list_of_fields_truncated_by_zip. Retrieved 2/10 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = var_2.serialize(var_0)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 'string'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = var_2.serialize(var_8)
    var_10 = bool(var_9 == var_8)
    assert var_10 is True

def test_case_0():
    var_0 = '1'
    var_1 = '2'
    var_2 = '3'
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = '1'
    var_1 = 'a'
    var_2 = [var_0, var_1]
    var_3 = '2'
    var_4 = 'b'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]
    var_7 = 1
    var_8 = [var_7]
    var_9 = [var_1]
    var_10 = [var_8, var_9]

def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 10
    var_1 = [var_0]



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'Alpha'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'Beta'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = 'c'
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'type'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = 'allow_null'
    var_4 = {var_3: var_1}
    var_5 = module_0.Choice(choices=var_2, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 is True
    var_7 = 1
    var_8 = var_5.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'key'
    var_4 = 'val'
    var_5 = {var_3: var_4}
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = [var_0, var_1]
    var_12 = var_10.validate(var_11)
    var_13 = bool(var_12 == [1, 2])
    assert var_13 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = 'content'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'content'



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'email'
    var_1 = {}
    var_2 = module_0.String(format=var_0, **var_1)
    var_3 = 'test@example.com'
    var_4 = var_2.validate(var_3)
    assert var_4 == 'test@example.com'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_serialize_with_list_of_items. Retrieved 5/15 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 10
    var_3 = 20
    var_4 = [var_2, var_3]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_array_constructor_with_default_value. Retrieved 4/5 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test Array'
    var_1 = 'A test array'
    var_2 = 2
    var_3 = 5
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_0, var_5: var_1}
    var_7 = module_0.Array(min_items=var_2, max_items=var_3, **var_6)
    var_8 = var_7.title
    assert var_8 == 'Test Array'
    var_9 = var_7.description
    assert var_9 == 'A test array'
    var_10 = var_7.min_items
    assert var_10 == 2
    var_11 = var_7.max_items
    assert var_11 == 5
    var_12 = var_7.items
    assert var_12 is None
    var_13 = var_7.additional_items
    assert var_13 is False
    var_14 = var_7.unique_items
    assert var_14 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Item'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = [var_1, var_1]
    var_3 = False
    var_4 = {}
    var_5 = module_0.Array(var_2, var_3, **var_4)
    var_6 = var_5.items
    var_7 = bool(var_5.items == [var_1, var_1])
    assert var_7 is True
    var_8 = var_5.min_items
    assert var_8 == 2
    var_9 = var_5.max_items
    assert var_9 == 2

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = var_2.min_items
    assert var_3 == 3
    var_4 = var_2.max_items
    assert var_4 == 3

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Extra'
    var_1 = module_0.Field(title=var_0, description=var_0)
    var_2 = None
    var_3 = {}
    var_4 = module_0.Array(var_2, var_1, **var_3)
    var_5 = var_4.additional_items
    var_6 = bool(var_4.additional_items == var_1)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = var_2.unique_items
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'default'
    var_4 = {var_3: var_2}
    var_5 = module_0.Array(**var_4)
    var_6 = var_5.default
    var_7 = bool(var_5.default == [1, 2])
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = var_3.allow_null
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_serialize_with_list_of_items. Retrieved 6/14 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]
    var_3 = 'serialized_1'
    var_4 = 'serialized_2'
    var_5 = [var_3, var_4]



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'Name'
    var_2 = module_0.Field(title=var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = 'User'
    var_6 = 'User object'
    var_7 = 'title'
    var_8 = 'description'
    var_9 = {var_7: var_5, var_8: var_6}
    var_10 = module_0.Object(properties=var_3, required=var_4, **var_9)
    var_11 = var_10.properties
    var_12 = bool(var_10.properties == var_3)
    assert var_12 is True
    var_13 = var_10.required
    var_14 = bool(var_10.required == ['name'])
    assert var_14 is True
    var_15 = var_10.title
    assert var_15 == 'User'
    var_16 = var_10.description
    assert var_16 == 'User object'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Extra'
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = module_0.Object(additional_properties=var_1, **var_2)
    var_4 = 'addly_prop'
    var_5 = locals()
    var_6 = var_4 in var_5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 5
    var_2 = {}
    var_3 = module_0.Object(min_properties=var_0, max_properties=var_1, **var_2)
    var_4 = var_3.min_properties
    assert var_4 == 1
    var_5 = var_3.max_properties
    assert var_5 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^user_'
    var_1 = 'User Pattern'
    var_2 = module_0.Field(title=var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = var_5.pattern_properties
    var_7 = bool(var_5.pattern_properties == var_3)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'not a field'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = bool(False)
    assert var_5 is True
    var_6 = bool(True)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Object(properties=var_0, **var_1)
    var_3 = var_2.properties
    var_4 = bool(var_2.properties == {})
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Object(required=var_0, **var_1)
    var_3 = var_2.required
    var_4 = bool(var_2.required == [])
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_validate_no_exception_on_valid_string_conversion. Retrieved 4/5 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123.45'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #15
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_serialize_with_items_as_list_skips_none_check. Retrieved 4/13 statements.


def test_case_0():
    var_0 = None
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_validate_union_predicate_true_via_multiple_messages. Retrieved 5/14 statements.
# Partially parsed test_validate_union_predicate_true_via_wrong_error_code. Retrieved 4/12 statements.
# Partially parsed test_validate_union_predicate_true_via_error_index. Retrieved 5/14 statements.


def test_case_0():
    var_0 = 'err1'
    var_1 = 'err2'
    var_2 = None
    var_3 = False
    var_4 = 'some_value'

def test_case_0():
    var_0 = 'not_type'
    var_1 = None
    var_2 = False
    var_3 = 'some_value'

def test_case_0():
    var_0 = 'type'
    var_1 = 0
    var_2 = None
    var_3 = False
    var_4 = 'some_value'



# Parsed testcases at query #18
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = False
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True
    var_5 = 'TRUE'
    var_6 = var_2.validate(var_5)
    assert var_6 is True
    var_7 = 'on'
    var_8 = var_2.validate(var_7)
    assert var_8 is True
    var_9 = '1'
    var_10 = var_2.validate(var_9)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False
    var_5 = 'OFF'
    var_6 = var_2.validate(var_5)
    assert var_6 is False
    var_7 = '0'
    var_8 = var_2.validate(var_7)
    assert var_8 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True
    var_4 = 0
    var_5 = var_2.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = ''
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = 'null'
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = 'none'
    var_9 = var_3.validate(var_8)
    assert var_9 is None
    var_10 = ''
    var_11 = var_3.validate(var_10)
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'not_a_boolean'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = True
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = 'type'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_validate_returns_validated_value_when_child_matches. Retrieved 1/23 statements.


def test_case_0():
    var_0 = 'success'



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'true'
    var_5 = var_3.validate(var_4)



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Username'
    var_1 = "The user's unique name"
    var_2 = True
    var_3 = False
    var_4 = 20
    var_5 = 3
    var_6 = '^[a-z]+$'
    var_7 = 'email'
    var_8 = 'title'
    var_9 = 'description'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_5, pattern=var_6, format=var_7, coerce_types=var_3, **var_10)
    var_12 = var_11.title
    assert var_12 == 'Username'
    var_13 = var_11.description
    assert var_13 == "The user's unique name"
    var_14 = var_11.allow_blank
    assert var_14 is True
    var_15 = var_11.trim_whitespace
    assert var_15 is False
    var_16 = var_11.max_length
    assert var_16 == 20
    var_17 = var_11.min_length
    assert var_17 == 3
    var_18 = var_11.pattern
    assert var_18 == '^[a-z]+$'
    var_19 = var_11.format
    assert var_19 == 'email'
    var_20 = var_11.coerce_types
    assert var_20 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = var_1.allow_blank
    assert var_2 is False
    var_3 = var_1.trim_whitespace
    assert var_3 is True
    var_4 = var_1.max_length
    assert var_4 is None
    var_5 = var_1.min_length
    assert var_5 is None
    var_6 = var_1.pattern
    assert var_6 is None
    var_7 = var_1.format
    assert var_7 is None
    var_8 = var_1.coerce_types
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = var_2.default
    assert var_3 == ''

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '\\d+'
    var_1 = module_0.compile(var_0)
    var_2 = {}
    var_3 = module_1.String(pattern=var_1, **var_2)
    var_4 = var_3.pattern
    assert var_4 == '\\d+'
    var_5 = var_3.pattern_regex
    var_6 = bool(var_3.pattern_regex == var_3.pattern_regex)
    assert var_6 is True

import typesystem.fields as module_0
import builtins as module_1

def test_case_0():
    var_0 = 'not_an_int'
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'Should have raised AssertionError for invalid max_length type'
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_1.Exception(*var_4, **var_5)
    var_7 = 1
    var_8 = 2
    var_9 = [var_7, var_8]
    var_10 = {}
    var_11 = module_0.String(min_length=var_9, **var_10)
    var_12 = 'Should have raised AssertionError for invalid min_length type'
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Exception(*var_13, **var_14)
    var_16 = 123
    var_17 = {}
    var_18 = module_0.String(pattern=var_16, **var_17)
    var_19 = 'Should have raised AssertionError for invalid pattern type'
    var_20 = [var_19]
    var_21 = {}
    var_22 = module_1.Exception(*var_20, **var_21)
    var_23 = True
    var_24 = {}
    var_25 = module_0.String(format=var_23, **var_24)
    var_26 = 'Should have raised AssertionError for invalid format type'
    var_27 = [var_26]
    var_28 = {}
    var_29 = module_1.Exception(*var_27, **var_28)



# Parsed testcases at query #22
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Prop'
    var_1 = 'Desc'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = 'Pattern'
    var_4 = module_0.Field(title=var_3, description=var_3)
    var_5 = 'Additional'
    var_6 = module_0.Field(title=var_5, description=var_5)
    var_7 = 'name'
    var_8 = 'age'
    var_9 = [var_7, var_8]
    var_10 = {var_7: var_2, var_8: var_2}
    var_11 = '^id_'
    var_12 = {var_11: var_4}
    var_13 = 2
    var_14 = 5
    var_15 = 'Test Object'
    var_16 = 'Test Description'
    var_17 = 'title'
    var_18 = 'description'
    var_19 = {var_17: var_15, var_18: var_16}
    var_20 = module_0.Object(properties=var_10, pattern_properties=var_12, additional_properties=var_6, property_names=var_2, min_properties=var_13, max_properties=var_14, required=var_9, **var_19)
    var_21 = var_20.properties
    var_22 = bool(var_20.properties == {'name': var_2, 'age': var_2})
    assert var_22 is True
    var_23 = var_20.pattern_properties
    var_24 = bool(var_20.pattern_properties == {'^id_': var_4})
    assert var_24 is True
    var_25 = var_20.additional_properties
    var_26 = bool(var_20.additional_properties == var_6)
    assert var_26 is True
    var_27 = var_20.property_names
    var_28 = bool(var_20.property_names == var_2)
    assert var_28 is True
    var_29 = var_20.min_properties
    assert var_29 == 2
    var_30 = var_20.max_properties
    assert var_30 == 5
    var_31 = var_20.required
    var_32 = bool(var_20.required == ['name', 'age'])
    assert var_32 is True
    var_33 = var_20.title
    assert var_33 == 'Test Object'
    var_34 = var_20.description
    assert var_34 == 'Test Description'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = var_2.additional_properties
    assert var_3 is True
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(additional_properties=var_4, **var_5)
    var_7 = var_6.additional_properties
    assert var_7 is False
    var_8 = None
    var_9 = {}
    var_10 = module_0.Object(additional_properties=var_8, **var_9)
    var_11 = var_10.additional_properties
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Single'
    var_1 = 'Desc'
    var_2 = module_0.Field(title=var_0, description=var_1)
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = var_4.additional_properties
    var_6 = bool(var_4.additional_properties == var_2)
    assert var_6 is True
    var_7 = var_4.properties
    var_8 = bool(var_4.properties == {})
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = var_1.properties
    var_3 = bool(var_1.properties == {})
    assert var_3 is True
    var_4 = var_1.pattern_properties
    var_5 = bool(var_1.pattern_properties == {})
    assert var_5 is True
    var_6 = var_1.additional_properties
    assert var_6 is True
    var_7 = var_1.property_names
    assert var_7 is None
    var_8 = var_1.min_properties
    assert var_8 is None
    var_9 = var_1.max_properties
    assert var_9 is None
    var_10 = var_1.required
    var_11 = bool(var_1.required == [])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'key'
    var_1 = 'not a field'
    var_2 = {var_0: var_1}
    var_3 = {}
    var_4 = module_0.Object(properties=var_2, **var_3)
    var_5 = 123
    var_6 = module_0.Field()
    var_7 = {var_5: var_6}
    var_8 = {}
    var_9 = module_0.Object(pattern_properties=var_7, **var_8)
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = {}
    var_15 = module_0.Object(required=var_13, **var_14)



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_array_validate_exact_items_success. Retrieved 3/16 statements.
# Partially parsed test_array_validate_exact_items_error. Retrieved 3/17 statements.
# Partially parsed test_array_validate_min_items_error. Retrieved 3/16 statements.
# Partially parsed test_array_validate_max_items_error. Retrieved 4/17 statements.
# Partially parsed test_array_validate_unique_items_error. Retrieved 4/16 statements.
# Partially parsed test_array_validate_item_validation_error. Retrieved 3/20 statements.
# Partially parsed test_array_validate_additional_items_field. Retrieved 3/23 statements.
# Partially parsed test_array_validate_empty_min_items_one. Retrieved 2/15 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = 'type'

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_1, var_0]

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_1]
    var_3 = 'exact_items'

def test_case_0():
    var_0 = 2
    var_1 = 1
    var_2 = [var_1]
    var_3 = 'min_items'

def test_case_0():
    var_0 = 1
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2]
    var_4 = 'max_items'

def test_case_0():
    var_0 = True
    var_1 = 1
    var_2 = 2
    var_3 = [var_1, var_2, var_1]
    var_4 = 'unique_items'

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = 'a'
    var_2 = [var_0, var_1]

def test_case_0():
    var_0 = 1
    var_1 = []
    var_2 = 'empty'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_validate_union_predicate_true_via_multiple_messages. Retrieved 1/11 statements.
# Partially parsed test_validate_union_predicate_true_via_wrong_code. Retrieved 1/12 statements.
# Partially parsed test_validate_union_predicate_true_via_index_present. Retrieved 1/12 statements.


def test_case_0():
    var_0 = 'some_value'

def test_case_0():
    var_0 = 'some_value'

def test_case_0():
    var_0 = 'some_value'



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_serialize_integer. Retrieved 4/5 statements.
# Partially parsed test_serialize_float. Retrieved 4/5 statements.
# Partially parsed test_serialize_string_numeric. Retrieved 4/5 statements.
# Partially parsed test_serialize_decimal_object. Retrieved 2/7 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = None
    var_3 = var_1.serialize(var_2)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = 10
    var_3 = var_1.serialize(var_2)
    var_4 = bool(var_3 == 10.0)
    assert var_4 is True
    var_5 = var_1.serialize(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = 10.5
    var_3 = var_1.serialize(var_2)
    var_4 = bool(var_3 == 10.5)
    assert var_4 is True
    var_5 = var_1.serialize(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '123.45'
    var_3 = var_1.serialize(var_2)
    var_4 = bool(var_3 == 123.45)
    assert var_4 is True
    var_5 = var_1.serialize(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Decimal(**var_0)
    var_2 = '0.1'



# Parsed testcases at query #26
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Field'
    var_5 = 'title'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = var_7.choices
    var_9 = bool(var_7.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')])
    assert var_9 is True
    var_10 = var_7.title
    assert var_10 == 'Test Field'
    var_11 = var_7.coerce_types
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '1'
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = {}
    var_8 = module_0.Choice(choices=var_6, **var_7)
    var_9 = var_8.choices
    var_10 = bool(var_8.choices == [('1', 'One'), ('2', 'Two')])
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'Desc'
    var_3 = True
    var_4 = 'description'
    var_5 = 'allow_null'
    var_6 = 'read_only'
    var_7 = {var_4: var_2, var_5: var_3, var_6: var_3}
    var_8 = module_0.Choice(choices=var_1, **var_7)
    var_9 = var_8.description
    assert var_9 == 'Desc'
    var_10 = var_8.allow_null
    assert var_10 is True
    var_11 = var_8.read_only
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Choice(choices=var_1, coerce_types=var_2, **var_3)
    var_5 = var_4.coerce_types
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'only_one'
    var_1 = (var_0,)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Choice(choices=var_2, **var_3)



# Parsed testcases at query #27
#--------------------------




import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'allow_null'
    var_3 = 'validate_or_error'
    var_4 = True
    var_5 = None
    var_6 = (var_5, var_5)
    var_7 = lambda self, v: var_6
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Union(var_13, **var_14)
    var_16 = var_15.validate(var_5)
    assert var_16 is None

import builtins as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = 'Field'
    var_1 = ()
    var_2 = 'allow_null'
    var_3 = 'validate_or_error'
    var_4 = False
    var_5 = None
    var_6 = (var_5, var_5)
    var_7 = lambda self, v: var_6
    var_8 = {var_2: var_4, var_3: var_7}
    var_9 = [var_0, var_1, var_8]
    var_10 = {}
    var_11 = module_0.type(*var_9, **var_10)
    var_12 = var_11()
    var_13 = [var_12]
    var_14 = {}
    var_15 = module_1.Union(var_13, **var_14)
    var_16 = None
    var_17 = var_15.validate(var_16)



# Parsed testcases at query #28
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = False
    var_2 = None
    var_3 = {}
    var_4 = module_0.Array(var_2, var_1, max_items=var_0, **var_3)
    var_5 = var_4.max_items
    assert var_5 == 5



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'expected'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 'expected'

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'expected'
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = 'unexpected'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = 'not_none'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = {}
    var_2 = module_0.Const(var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 123



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_validate_numeric_type_int_float_non_integer_triggers_line_11. Retrieved 4/9 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = 'integer'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_object_validate_invalid_key_type. Retrieved 4/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 1
    var_3 = 2
    var_4 = 3
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}



####################################################################
#        TEST GENERATION BEGINS (DEEPMOSA + Gemma 4 t=0.8)         #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_validate_valid_float. Retrieved 5/6 statements.
# Partially parsed test_validate_string_conversion. Retrieved 4/5 statements.
# Partially parsed test_validate_exclusive_minimum_constraint. Retrieved 7/9 statements.
# Partially parsed test_validate_exclusive_maximum_constraint. Retrieved 7/9 statements.
# Partially parsed test_validate_multiple_of_float. Retrieved 7/9 statements.
# Partially parsed test_validate_precision. Retrieved 2/4 statements.
# Partially parsed test_validate_numeric_type_cast. Retrieved 1/3 statements.
# Partially parsed test_validate_integer_error_on_float_type_int. Retrieved 1/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10.5
    var_4 = var_2.validate(var_3)
    var_5 = '10.5'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '123.45'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(coerce_types=var_0, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Number(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'May not be null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = True
    var_3 = var_1.validate(var_2)
    var_4 = 'Must be a number'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 5
    var_4 = 4
    var_5 = var_2.validate(var_4)
    var_6 = 'Must be greater than or equal to 5'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 5.1
    var_4 = var_2.validate(var_3)
    var_5 = '5.1'
    var_6 = 5
    var_7 = var_2.validate(var_6)
    var_8 = 'Must be greater than 5'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10
    var_4 = 11
    var_5 = var_2.validate(var_4)
    var_6 = 'Must be less than or equal to 10'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9.9
    var_4 = var_2.validate(var_3)
    var_5 = '9.9'
    var_6 = 10
    var_7 = var_2.validate(var_6)
    var_8 = 'Must be less than 10'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    assert var_4 == 4
    var_5 = 3
    var_6 = var_2.validate(var_5)
    var_7 = 'Must be a multiple of 2'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 0.5
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 1.5
    var_4 = var_2.validate(var_3)
    var_5 = '1.5'
    var_6 = 1.2
    var_7 = var_2.validate(var_6)
    var_8 = 'Must be a multiple of 0.5'

def test_case_0():
    var_0 = '0.01'
    var_1 = '1.23456'

def test_case_0():
    var_0 = 10.0

def test_case_0():
    var_0 = 10.5
    var_1 = 'Must be an integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'not-a-number'
    var_3 = var_1.validate(var_2)
    var_4 = 'Must be a number'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = 'Must be finite'



# Parsed testcases at query #2
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = {}
    var_3 = module_0.String(max_length=var_0, min_length=var_1, **var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  trimmed  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'trimmed'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  not trimmed  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  not trimmed  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcdef'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '12345'
    var_4 = var_2.validate(var_3)
    assert var_4 == '12345'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc123'
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #3
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = False
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True
    var_5 = 'TRUE'
    var_6 = var_2.validate(var_5)
    assert var_6 is True
    var_7 = 'on'
    var_8 = var_2.validate(var_7)
    assert var_8 is True
    var_9 = '1'
    var_10 = var_2.validate(var_9)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False
    var_5 = 'off'
    var_6 = var_2.validate(var_5)
    assert var_6 is False
    var_7 = '0'
    var_8 = var_2.validate(var_7)
    assert var_8 is False
    var_9 = ''
    var_10 = var_2.validate(var_9)
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True
    var_4 = 0
    var_5 = var_2.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a boolean.'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'May not be null.'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = 'null'
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = 'none'
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = ''
    var_9 = var_3.validate(var_8)
    assert var_9 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'not_a_boolean_value'
    var_4 = var_2.validate(var_3)
    var_5 = 'Must be a boolean.'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'list'
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = 'Must be a boolean.'



# Parsed testcases at query #4
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_array_serialize_with_single_item_type. Retrieved 4/10 statements.
# Partially parsed test_array_serialize_with_list_of_item_types. Retrieved 7/18 statements.
# Partially parsed test_array_serialize_with_additional_items_field. Retrieved 7/14 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Array(var_0, **var_3)
    var_5 = var_4.serialize(var_0)
    assert var_5 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Array(var_0, **var_1)
    var_3 = 1
    var_4 = 'string'
    var_5 = 'key'
    var_6 = 'value'
    var_7 = {var_5: var_6}
    var_8 = [var_3, var_4, var_7]
    var_9 = var_2.serialize(var_8)
    var_10 = bool(var_9 == [1, 'string', {'key': 'value'}])
    assert var_10 is True

def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]

def test_case_0():
    var_0 = 10
    var_1 = 'hello'
    var_2 = [var_0, var_1]
    var_3 = 20
    var_4 = 'world'
    var_5 = [var_3, var_4]
    var_6 = [var_2, var_5]

def test_case_0():
    var_0 = 1
    var_1 = [var_0]
    var_2 = 2
    var_3 = 3
    var_4 = 4
    var_5 = [var_2, var_3, var_4]
    var_6 = [var_1, var_5]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_object_validate_property_names_constraint. Retrieved 9/20 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Object(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 'not'
    var_3 = 'a'
    var_4 = 'dict'
    var_5 = [var_2, var_3, var_4]
    var_6 = var_1.validate(var_5)
    var_7 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Object(**var_0)
    var_2 = 123
    var_3 = 'value'
    var_4 = {var_2: var_3}
    var_5 = var_1.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = [var_0]
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, required=var_4, **var_5)
    var_7 = {}
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 1
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(max_properties=var_0, **var_1)
    var_3 = 'a'
    var_4 = 'b'
    var_5 = 1
    var_6 = 2
    var_7 = {var_3: var_5, var_4: var_6}
    var_8 = var_2.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = 'age'
    var_2 = {}
    var_3 = module_0.String(**var_2)
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {var_0: var_3, var_1: var_5}
    var_7 = {}
    var_8 = module_0.Object(properties=var_6, **var_7)
    var_9 = 'extra'
    var_10 = 'John'
    var_11 = 30
    var_12 = 'allowed'
    var_13 = {var_0: var_10, var_1: var_11, var_9: var_12}
    var_14 = var_8.validate(var_13)
    var_15 = bool(var_14 == {'name': 'John', 'age': 30, 'extra': 'allowed'})
    assert var_15 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = False
    var_5 = {}
    var_6 = module_0.Object(properties=var_3, additional_properties=var_4, **var_5)
    var_7 = 'a'
    var_8 = 'b'
    var_9 = 'val'
    var_10 = 'not_allowed'
    var_11 = {var_7: var_9, var_8: var_10}
    var_12 = var_6.validate(var_11)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Integer(**var_4)
    var_6 = {}
    var_7 = module_0.Object(properties=var_3, additional_properties=var_5, **var_6)
    var_8 = 'b'
    var_9 = 'val'
    var_10 = 123
    var_11 = {var_0: var_9, var_8: var_10}
    var_12 = var_7.validate(var_11)
    var_13 = bool(var_12 == {'a': 'val', 'b': 123})
    assert var_13 is True
    var_14 = 'a'
    var_15 = 'b'
    var_16 = 'val'
    var_17 = 'not_an_int'
    var_18 = {var_14: var_16, var_15: var_17}
    var_19 = var_7.validate(var_18)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = 'allowed'
    var_5 = 'val'
    var_6 = {var_4: var_5}
    var_7 = 'allowed'
    var_8 = 'forbidden'
    var_9 = 'val'
    var_10 = {var_8: var_9}



# Parsed testcases at query #7
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '^user_'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(pattern_properties=var_3, **var_4)
    var_6 = 'user_id'
    var_7 = 'other'
    var_8 = '123'
    var_9 = 'data'
    var_10 = {var_6: var_8, var_7: var_9}
    var_11 = var_5.validate(var_10)
    var_12 = bool(var_11 == {'user_id': '123', 'other': 'data'})
    assert var_12 is True



# Parsed testcases at query #8
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Number(precision=var_0, **var_3)
    var_5 = 1.2345
    var_6 = var_4.validate(var_5)
    var_7 = bool(var_6 == 1.23)
    assert var_7 is True



# Parsed testcases at query #9
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = 'not a list'
    var_3 = var_1.validate(var_2)
    var_4 = 'Must be an array.'
    var_5 = bool(False)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Array(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    var_6 = 'May not be null.'
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = 'Must have at least 3 items.'
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(max_items=var_0, **var_1)
    var_3 = 1
    var_4 = 2
    var_5 = [var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = 'Must have no more than 1 items.'
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Array(exact_items=var_0, **var_1)
    var_3 = 1
    var_4 = [var_3]
    var_5 = var_2.validate(var_4)
    var_6 = 'Must have 2 items.'
    var_7 = bool(False)
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Array(min_items=var_0, **var_1)
    var_3 = []
    var_4 = var_2.validate(var_3)
    var_5 = 'Must not be empty.'
    var_6 = bool(False)
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = 'valid'
    var_5 = 123
    var_6 = [var_4, var_5]
    var_7 = var_3.validate(var_6)
    var_8 = bool(False)
    assert var_8 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(var_1, unique_items=var_2, **var_3)
    var_5 = 'a'
    var_6 = 'b'
    var_7 = [var_5, var_6, var_5]
    var_8 = var_4.validate(var_7)
    var_9 = bool(False)
    assert var_9 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = {}
    var_6 = module_0.Array(var_4, **var_5)
    var_7 = 'hello'
    var_8 = 42
    var_9 = [var_7, var_8]
    var_10 = var_6.validate(var_9)
    var_11 = bool(var_10 == ['hello', 42])
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = [var_1]
    var_3 = {}
    var_4 = module_0.Integer(**var_3)
    var_5 = {}
    var_6 = module_0.Array(var_2, var_4, **var_5)
    var_7 = 'first'
    var_8 = 10
    var_9 = 20
    var_10 = [var_7, var_8, var_9]
    var_11 = var_6.validate(var_10)
    var_12 = bool(var_11 == ['first', 10, 20])
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = {}
    var_3 = module_0.Integer(**var_2)
    var_4 = [var_1, var_3]
    var_5 = False
    var_6 = {}
    var_7 = module_0.Array(var_4, var_5, **var_6)
    var_8 = 'first'
    var_9 = 1
    var_10 = 2
    var_11 = [var_8, var_9, var_10]
    var_12 = var_7.validate(var_11)
    var_13 = 'first'
    var_14 = 1
    var_15 = [var_13, var_14]
    var_16 = var_7.validate(var_15)
    var_17 = bool(var_16 == ['first', 1])
    assert var_17 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_object_property_names_validation_passes. Retrieved 5/12 statements.


def test_case_0():
    var_0 = 'valid_key'
    var_1 = None
    var_2 = 'test_key'
    var_3 = 'some_value'
    var_4 = {var_2: var_3}



# Parsed testcases at query #11
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = {}
    var_3 = module_0.String(allow_blank=var_0, trim_whitespace=var_1, **var_2)
    var_4 = '  not empty  '
    var_5 = var_3.validate(var_4)
    assert var_5 == 'not empty'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = {}
    var_3 = module_0.String(allow_blank=var_0, trim_whitespace=var_1, **var_2)
    var_4 = ''
    var_5 = var_3.validate(var_4)
    assert var_5 == ''



# Parsed testcases at query #12
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = 'invalid_value'
    var_6 = var_4.validate(var_5)



# Parsed testcases at query #13
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Username'
    var_1 = "The user's unique name"
    var_2 = True
    var_3 = False
    var_4 = 20
    var_5 = 3
    var_6 = '^\\w+$'
    var_7 = 'email'
    var_8 = 'title'
    var_9 = 'description'
    var_10 = {var_8: var_0, var_9: var_1}
    var_11 = module_0.String(allow_blank=var_2, trim_whitespace=var_3, max_length=var_4, min_length=var_5, pattern=var_6, format=var_7, coerce_types=var_3, **var_10)
    var_12 = var_11.title
    assert var_12 == 'Username'
    var_13 = var_11.description
    assert var_13 == "The user's unique name"
    var_14 = var_11.allow_blank
    assert var_14 is True
    var_15 = var_11.trim_whitespace
    assert var_15 is False
    var_16 = var_11.max_length
    assert var_16 == 20
    var_17 = var_11.min_length
    assert var_17 == 3
    var_18 = var_11.pattern
    assert var_18 == '^\\w+$'
    var_19 = var_11.format
    assert var_19 == 'email'
    var_20 = var_11.coerce_types
    assert var_20 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = var_5.allow_blank
    assert var_6 is False
    var_7 = var_5.trim_whitespace
    assert var_7 is True
    var_8 = var_5.max_length
    assert var_8 is None
    var_9 = var_5.min_length
    assert var_9 is None
    var_10 = var_5.pattern
    assert var_10 is None
    var_11 = var_5.format
    assert var_11 is None
    var_12 = var_5.coerce_types
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'Test'
    var_1 = 'Desc'
    var_2 = True
    var_3 = 'title'
    var_4 = 'description'
    var_5 = {var_3: var_0, var_4: var_1}
    var_6 = module_0.String(allow_blank=var_2, **var_5)
    var_7 = var_6.default
    assert var_7 == ''

import re as module_0
import typesystem.fields as module_1

def test_case_0():
    var_0 = '[a-z]+'
    var_1 = module_0.compile(var_0)
    var_2 = 'Test'
    var_3 = 'Desc'
    var_4 = 'title'
    var_5 = 'description'
    var_6 = {var_4: var_2, var_5: var_3}
    var_7 = module_1.String(pattern=var_1, **var_6)
    var_8 = var_7.pattern
    assert var_8 == '[a-z]+'
    var_9 = var_7.pattern_regex
    var_10 = bool(var_7.pattern_regex == var_1)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 123
    var_1 = 'Desc'
    var_2 = 'title'
    var_3 = 'description'
    var_4 = {var_2: var_0, var_3: var_1}
    var_5 = module_0.String(**var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = bool(True)
    assert var_7 is True
    var_8 = 'Test'
    var_9 = 'Desc'
    var_10 = 'not_an_int'
    var_11 = 'title'
    var_12 = 'description'
    var_13 = {var_11: var_8, var_12: var_9}
    var_14 = module_0.String(max_length=var_10, **var_13)
    var_15 = bool(False)
    assert var_15 is True
    var_16 = bool(True)
    assert var_16 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_object_validate_not_null_and_is_dict. Retrieved 7/9 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = {}
    var_2 = module_0.String(**var_1)
    var_3 = {var_0: var_2}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = 'test'
    var_7 = {var_0: var_6}
    var_8 = var_5.validate(var_7)
    var_9 = bool(var_8 == {'name': 'test'})
    assert var_9 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_serialize_with_list_of_items. Retrieved 6/15 statements.


def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = 'serialized_a'
    var_4 = 'serialized_b'
    var_5 = [var_3, var_4]



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_array_validate_null_error. Retrieved 2/10 statements.


def test_case_0():
    var_0 = False
    var_1 = None
    var_2 = 'May not be null.'

def test_case_0():
    pass



# Parsed testcases at query #17
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'val1'
    var_2 = (var_0, var_1)
    var_3 = 'b'
    var_4 = 'val2'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = False
    var_8 = 'allow_null'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.validate(var_0)
    assert var_11 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = 'c'
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, coerce_types=var_3, **var_5)
    var_7 = ''
    var_8 = var_6.validate(var_7)
    assert var_8 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = True
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = False
    var_5 = 'allow_null'
    var_6 = {var_5: var_3}
    var_7 = module_0.Choice(choices=var_2, coerce_types=var_4, **var_6)
    var_8 = ''
    var_9 = var_7.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = [var_0, var_1]
    var_3 = 'allow_null'
    var_4 = {var_3: var_1}
    var_5 = module_0.Choice(choices=var_2, **var_4)
    var_6 = var_5.validate(var_0)
    assert var_6 is True
    var_7 = var_5.validate(var_1)
    assert var_7 is False
    var_8 = 1
    var_9 = var_5.validate(var_8)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = 0
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = var_6.validate(var_0)
    assert var_7 == 1
    var_8 = var_6.validate(var_3)
    assert var_8 == 0
    var_9 = True
    var_10 = var_6.validate(var_9)



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_array_constructor_with_list_of_fields. Retrieved 5/8 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Array(**var_0)
    var_2 = var_1.items
    assert var_2 is None
    var_3 = var_1.additional_items
    assert var_3 is False
    var_4 = var_1.min_items
    assert var_4 is None
    var_5 = var_1.max_items
    assert var_5 is None
    var_6 = var_1.unique_items
    assert var_6 is False
    var_7 = var_1.title
    assert var_7 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'test'
    var_1 = module_0.Field(title=var_0)
    var_2 = {}
    var_3 = module_0.Array(var_1, **var_2)
    var_4 = var_3.items
    var_5 = bool(var_3.items == var_1)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = False
    var_3 = 'f2'
    var_4 = var_3 if var_2 else var_3

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 5
    var_3 = {}
    var_4 = module_0.Array(var_1, exact_items=var_2, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 5
    var_6 = var_4.max_items
    assert var_6 == 5

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'f2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1]
    var_5 = {}
    var_6 = module_0.Array(var_4, var_3, **var_5)
    var_7 = var_6.items
    var_8 = bool(var_6.items == [var_1])
    assert var_8 is True
    var_9 = var_6.additional_items
    var_10 = bool(var_6.additional_items == var_3)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = 10
    var_2 = True
    var_3 = {}
    var_4 = module_0.Array(min_items=var_0, max_items=var_1, unique_items=var_2, **var_3)
    var_5 = var_4.min_items
    assert var_5 == 2
    var_6 = var_4.max_items
    assert var_6 == 10
    var_7 = var_4.unique_items
    assert var_7 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'ArrayField'
    var_1 = 'Desc'
    var_2 = True
    var_3 = 'title'
    var_4 = 'description'
    var_5 = 'allow_null'
    var_6 = {var_3: var_0, var_4: var_1, var_5: var_2}
    var_7 = module_0.Array(**var_6)
    var_8 = var_7.title
    assert var_8 == 'ArrayField'
    var_9 = var_7.description
    assert var_9 == 'Desc'
    var_10 = var_7.allow_null
    assert var_10 is True



# Parsed testcases at query #19
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 1
    var_1 = {}
    var_2 = module_0.Object(min_properties=var_0, **var_1)
    var_3 = {}
    var_4 = var_2.validate(var_3)



# Parsed testcases at query #20
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = 1
    var_2 = {}
    var_3 = module_0.String(max_length=var_0, min_length=var_1, **var_2)
    var_4 = 'hello'
    var_5 = var_3.validate(var_4)
    assert var_5 == 'hello'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  trimmed  '
    var_4 = var_2.validate(var_3)
    assert var_4 == 'trimmed'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(trim_whitespace=var_0, **var_1)
    var_3 = '  not_trimmed  '
    var_4 = var_2.validate(var_3)
    assert var_4 == '  not_trimmed  '

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(**var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.String(coerce_types=var_1, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.String(allow_blank=var_0, coerce_types=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = '   '
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.String(allow_blank=var_0, **var_1)
    var_3 = ' '
    var_4 = var_2.validate(var_3)
    assert var_4 == ''

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.String(allow_blank=var_0, coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    assert var_6 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 123
    var_3 = var_1.validate(var_2)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.String(min_length=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = 3
    var_1 = {}
    var_2 = module_0.String(max_length=var_0, **var_1)
    var_3 = 'abcd'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = '123'
    var_4 = var_2.validate(var_3)
    assert var_4 == '123'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '^\\d+$'
    var_1 = {}
    var_2 = module_0.String(pattern=var_0, **var_1)
    var_3 = 'abc'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'hello\x00world'
    var_3 = var_1.validate(var_2)
    assert var_3 == 'helloworld'



# Parsed testcases at query #21
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'name'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Object(required=var_1, **var_2)
    var_4 = 'age'
    var_5 = 'John Doe'
    var_6 = 30
    var_7 = {var_0: var_5, var_4: var_6}
    var_8 = var_3.validate(var_7)
    var_9 = 'name'
    var_10 = bool('name' in var_8)
    assert var_10 is True
    var_11 = var_8['name']
    assert var_11 == 'John Doe'
    var_12 = var_8['age']
    assert var_12 == 30



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_validate_returns_none_when_value_is_none_and_allow_null_is_true. Retrieved 2/13 statements.


def test_case_0():
    var_0 = True
    var_1 = None

def test_case_0():
    pass



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_validate_raises_null_error_when_value_is_none_and_not_allowed. Retrieved 7/17 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = [var_0, var_1]
    var_3 = False
    var_4 = 'allow_null'
    var_5 = {var_4: var_3}
    var_6 = module_0.Choice(choices=var_2, **var_5)
    var_7 = None
    var_8 = var_6.validate(var_7)



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_object_additional_properties_not_field. Retrieved 6/10 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Object(additional_properties=var_0, **var_1)
    var_3 = 'extra'
    var_4 = 'value'
    var_5 = {var_3: var_4}
    var_6 = var_2.validate(var_5)



# Parsed testcases at query #25
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.String(**var_0)
    var_2 = 'name'
    var_3 = {var_2: var_1}
    var_4 = {}
    var_5 = module_0.Object(properties=var_3, **var_4)
    var_6 = {}
    var_7 = var_5.validate(var_6)
    var_8 = 'name'
    var_9 = bool('name' not in var_7)
    assert var_9 is True
    var_10 = bool(var_7 == {})
    assert var_10 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_choice_constructor_with_default. Retrieved 3/4 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = [var_0, var_1, var_2]
    var_4 = 'Test Field'
    var_5 = 'title'
    var_6 = {var_5: var_4}
    var_7 = module_0.Choice(choices=var_3, **var_6)
    var_8 = var_7.choices
    var_9 = bool(var_7.choices == [('a', 'a'), ('b', 'b'), ('c', 'c')])
    assert var_9 is True
    var_10 = var_7.title
    assert var_10 == 'Test Field'
    var_11 = var_7.coerce_types
    assert var_11 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = '1'
    var_1 = 'One'
    var_2 = (var_0, var_1)
    var_3 = '2'
    var_4 = 'Two'
    var_5 = (var_3, var_4)
    var_6 = [var_2, var_5]
    var_7 = 'Tuple Field'
    var_8 = 'title'
    var_9 = {var_8: var_7}
    var_10 = module_0.Choice(choices=var_6, **var_9)
    var_11 = var_10.choices
    var_12 = bool(var_10.choices == [('1', 'One'), ('2', 'Two')])
    assert var_12 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = 'default'
    var_3 = {var_2: var_0}
    var_4 = module_0.Choice(choices=var_1, **var_3)
    var_5 = var_4.default
    assert var_5 == 'a'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = True
    var_3 = 'allow_null'
    var_4 = {var_3: var_2}
    var_5 = module_0.Choice(choices=var_1, **var_4)
    var_6 = var_5.allow_null
    assert var_6 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Choice(choices=var_0, **var_1)
    var_3 = var_2.choices
    var_4 = bool(var_2.choices == [])
    assert var_4 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = [var_0]
    var_2 = False
    var_3 = {}
    var_4 = module_0.Choice(choices=var_1, coerce_types=var_2, **var_3)
    var_5 = var_4.coerce_types
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = 'a'
    var_1 = 'b'
    var_2 = 'c'
    var_3 = (var_0, var_1, var_2)
    var_4 = [var_3]
    var_5 = {}
    var_6 = module_0.Choice(choices=var_4, **var_5)



# Parsed testcases at query #27
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = True
    var_2 = 'allow_null'
    var_3 = {var_2: var_0}
    var_4 = module_0.Number(coerce_types=var_1, **var_3)
    var_5 = ''
    var_6 = var_4.validate(var_5)
    var_7 = bool(var_6 != None)
    assert var_7 is True



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_serialize_with_list_of_items. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = [var_0, var_1, var_2]



# Parsed testcases at query #29
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Array(unique_items=var_0, **var_1)
    var_3 = 2
    var_4 = 3
    var_5 = [var_0, var_3, var_4]
    var_6 = var_2.validate(var_5)
    var_7 = bool(var_6 == [1, 2, 3])
    assert var_7 is True



# Parsed testcases at query #30
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = False
    var_4 = var_2.validate(var_3)
    assert var_4 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    assert var_4 is True
    var_5 = 'TRUE'
    var_6 = var_2.validate(var_5)
    assert var_6 is True
    var_7 = 'on'
    var_8 = var_2.validate(var_7)
    assert var_8 is True
    var_9 = '1'
    var_10 = var_2.validate(var_9)
    assert var_10 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'false'
    var_4 = var_2.validate(var_3)
    assert var_4 is False
    var_5 = 'off'
    var_6 = var_2.validate(var_5)
    assert var_6 is False
    var_7 = '0'
    var_8 = var_2.validate(var_7)
    assert var_8 is False
    var_9 = ''
    var_10 = var_2.validate(var_9)
    assert var_10 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 is True
    var_4 = 0
    var_5 = var_2.validate(var_4)
    assert var_5 is False

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = 'allow_null'
    var_2 = {var_1: var_0}
    var_3 = module_0.Boolean(coerce_types=var_0, **var_2)
    var_4 = None
    var_5 = var_3.validate(var_4)
    assert var_5 is None
    var_6 = 'null'
    var_7 = var_3.validate(var_6)
    assert var_7 is None
    var_8 = 'none'
    var_9 = var_3.validate(var_8)
    assert var_9 is None
    var_10 = ''
    var_11 = var_3.validate(var_10)
    assert var_11 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = False
    var_2 = 'allow_null'
    var_3 = {var_2: var_1}
    var_4 = module_0.Boolean(coerce_types=var_0, **var_3)
    var_5 = None
    var_6 = var_4.validate(var_5)
    var_7 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = False
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'true'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Boolean(coerce_types=var_0, **var_1)
    var_3 = 'not_a_boolean'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_array_validate_unique_items_collision. Retrieved 5/12 statements.


def test_case_0():
    var_0 = True
    var_1 = 'apple'
    var_2 = [var_1, var_1]
    var_3 = 'ValidationError with unique_items code was not raised'
    var_4 = AssertionError(var_3)



# Parsed testcases at query #32
#--------------------------




import typesystem.fields as module_0

def test_case_0():
    var_0 = 'f1'
    var_1 = module_0.Field(title=var_0)
    var_2 = 'f2'
    var_3 = module_0.Field(title=var_2)
    var_4 = [var_1, var_3]
    var_5 = 5
    var_6 = {}
    var_7 = module_0.Array(var_4, max_items=var_5, **var_6)
    var_8 = var_7.max_items
    assert var_8 == 5



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_validate_valid_string_coercion. Retrieved 4/5 statements.
# Partially parsed test_validate_null_allowed. Retrieved 4/5 statements.
# Partially parsed test_validate_null_not_allowed. Retrieved 4/6 statements.
# Partially parsed test_validate_precision_rounding. Retrieved 5/6 statements.
# Partially parsed test_validate_numeric_type_int_constraint. Retrieved 2/6 statements.


import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10
    var_4 = var_2.validate(var_3)
    assert var_4 == 10

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 10.5
    var_4 = var_2.validate(var_3)
    var_5 = bool(var_4 == 10.5)
    assert var_5 is True

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = '10.5'
    var_4 = var_2.validate(var_3)

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    assert var_4 is None

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = None
    var_4 = var_2.validate(var_3)
    var_5 = 'null'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = True
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(minimum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 5
    var_4 = 4
    var_5 = var_2.validate(var_4)
    var_6 = 'minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 5
    var_1 = {}
    var_2 = module_0.Number(exclusive_minimum=var_0, **var_1)
    var_3 = 6
    var_4 = var_2.validate(var_3)
    assert var_4 == 6
    var_5 = 5
    var_6 = var_2.validate(var_5)
    var_7 = 'exclusive_minimum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(maximum=var_0, **var_1)
    var_3 = var_2.validate(var_0)
    assert var_3 == 10
    var_4 = 11
    var_5 = var_2.validate(var_4)
    var_6 = 'maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 10
    var_1 = {}
    var_2 = module_0.Number(exclusive_maximum=var_0, **var_1)
    var_3 = 9
    var_4 = var_2.validate(var_3)
    assert var_4 == 9
    var_5 = 10
    var_6 = var_2.validate(var_5)
    var_7 = 'exclusive_maximum'

import typesystem.fields as module_0

def test_case_0():
    var_0 = 2
    var_1 = {}
    var_2 = module_0.Number(multiple_of=var_0, **var_1)
    var_3 = 4
    var_4 = var_2.validate(var_3)
    assert var_4 == 4
    var_5 = 3
    var_6 = var_2.validate(var_5)
    var_7 = 'multiple_of'

import typesystem.fields as module_0

def test_case_0():
    var_0 = '0.01'
    var_1 = {}
    var_2 = module_0.Number(precision=var_0, **var_1)
    var_3 = '1.234'
    var_4 = var_2.validate(var_3)
    var_5 = '1.23'

def test_case_0():
    var_0 = 10.0
    var_1 = 10.5
    var_2 = 'integer'

import typesystem.fields as module_0

def test_case_0():
    var_0 = True
    var_1 = {}
    var_2 = module_0.Number(coerce_types=var_0, **var_1)
    var_3 = 'not-a-number'
    var_4 = var_2.validate(var_3)
    var_5 = 'type'

import typesystem.fields as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Number(**var_0)
    var_2 = 'inf'
    var_3 = float(var_2)
    var_4 = var_1.validate(var_3)
    var_5 = 'finite'



