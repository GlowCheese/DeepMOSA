####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_string_by_mask_default_mask. Retrieved 9/13 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 9/29 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 12/15 statements.
# Partially parsed test_generate_string_by_mask_fixed_characters. Retrieved 14/19 statements.
# Partially parsed test_generate_string_by_mask_only_char_placeholder. Retrieved 4/6 statements.
# Partially parsed test_generate_string_by_mask_only_digit_placeholder. Retrieved 4/6 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = var_1[var_3]
    var_6 = 1
    var_7 = 4
    var_8 = var_1[var_6:var_7]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##-@@##'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 9
    var_4 = '-'
    var_5 = 0
    var_6 = 4
    var_7 = 1
    var_8 = 2

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'AA99'
    var_2 = 'A'
    var_3 = '9'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = 2
    var_8 = var_4[var_6:var_7]
    var_9 = var_4[var_6:var_7]
    var_10 = 4
    var_11 = var_4[var_7:var_10]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'CODE-@##-END'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = 'CODE-'
    var_4 = '-END'
    var_5 = 5
    var_6 = 8
    var_7 = var_2[var_5:var_6]
    var_8 = 0
    var_9 = var_7[var_8]
    var_10 = var_7[var_8]
    var_11 = 1
    var_12 = 3
    var_13 = var_7[var_11:var_12]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'same placeholder'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = ''
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == ''

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 3

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '###'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 3

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'FIXED-STRING'
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == 'FIXED-STRING'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_float_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_str_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_bytes_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_bytearray_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_memoryview_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_empty_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_negative_int_seed. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = bool(var_0 is not None)
    assert var_1 is True

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = 3.14

def test_case_0():
    var_0 = 'test'

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = b'seed'

def test_case_0():
    var_0 = b'seed'
    var_1 = bytearray(var_0)

def test_case_0():
    var_0 = b'seed'
    var_1 = memoryview(var_0)

def test_case_0():
    var_0 = ''

def test_case_0():
    var_0 = -123



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_float_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_str_seed. Retrieved 2/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_inherits_from_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 5/7 statements.
# Partially parsed test_constructor_initializes_random_state. Retrieved 3/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = 1
    var_4 = 100

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = range(var_2)
    var_4 = 1
    var_5 = 100
    var_6 = [var_0.randint(var_4, var_5) for _ in var_3]
    var_7 = range(var_2)
    var_8 = [var_1.randint(var_4, var_5) for _ in var_7]
    var_9 = bool(var_6 != var_8)
    assert var_9 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_initializes_with_float_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_initializes_with_str_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_initializes_with_bytes_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_initializes_with_bytearray_seed. Retrieved 2/6 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

def test_case_0():
    var_0 = 42

def test_case_0():
    var_0 = None

def test_case_0():
    var_0 = 3.14

def test_case_0():
    var_0 = 'test_seed'

def test_case_0():
    var_0 = b'bytes_seed'

def test_case_0():
    var_0 = b'bytearray_seed'
    var_1 = bytearray(var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_float_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_str_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_bytes_seed. Retrieved 2/5 statements.
# Partially parsed test_constructor_initializes_with_bytearray_seed. Retrieved 3/6 statements.
# Partially parsed test_constructor_initializes_with_memoryview_seed. Retrieved 3/6 statements.
# Partially parsed test_constructor_initializes_with_version_arg. Retrieved 1/5 statements.
# Partially parsed test_constructor_initializes_with_int_seed_and_version. Retrieved 2/6 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = bytearray(var_0)
    var_2 = module_0.Random(var_1)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = memoryview(var_0)
    var_2 = module_0.Random(var_1)

def test_case_0():
    var_0 = 2

def test_case_0():
    var_0 = 42
    var_1 = 2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 5/7 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 4/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = 1
    var_3 = 100
    var_4 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)
    var_2 = 1
    var_3 = 10
    var_4 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = range(var_2)
    var_4 = 1
    var_5 = 1000
    var_6 = [var_0.randint(var_4, var_5) for _ in var_3]
    var_7 = range(var_2)
    var_8 = [var_1.randint(var_4, var_5) for _ in var_7]
    var_9 = bool(var_6 != var_8)
    assert var_9 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = 10
    var_4 = range(var_3)
    var_5 = [var_1.random() for _ in var_4]
    var_6 = range(var_3)
    var_7 = [var_2.random() for _ in var_6]
    var_8 = bool(var_5 == var_7)
    assert var_8 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 5/7 statements.
# Partially parsed test_constructor_inherits_from_random_random. Retrieved 1/3 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = 1
    var_4 = 100

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = range(var_2)
    var_4 = 1
    var_5 = 100
    var_6 = [var_0.randint(var_4, var_5) for _ in var_3]
    var_7 = range(var_2)
    var_8 = [var_1.randint(var_4, var_5) for _ in var_7]
    var_9 = bool(var_6 != var_8)
    assert var_9 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_default. Retrieved 1/2 statements.
# Partially parsed test_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_inherits_from_random. Retrieved 1/3 statements.
# Partially parsed test_constructor_no_args. Retrieved 3/4 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 4/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)
    var_2 = 1
    var_3 = 10
    var_4 = 1



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_constructor_without_seed. Retrieved 1/4 statements.
# Partially parsed test_constructor_with_seed. Retrieved 1/5 statements.
# Partially parsed test_constructor_inherits_random_methods. Retrieved 3/4 statements.
# Partially parsed test_constructor_initializes_custom_methods. Retrieved 3/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

def test_case_0():
    var_0 = 42

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 10
    var_3 = 1

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.randints()
    var_2 = len(var_1)
    assert var_2 == 3



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_string_by_mask_default_mask. Retrieved 12/17 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 16/22 statements.
# Partially parsed test_generate_string_by_mask_different_placeholders. Retrieved 16/22 statements.
# Partially parsed test_generate_string_by_mask_mixed_characters. Retrieved 11/14 statements.
# Partially parsed test_generate_string_by_mask_long_mask. Retrieved 9/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = var_1[var_3]
    var_6 = 1
    var_7 = var_1[var_6]
    var_8 = 2
    var_9 = var_1[var_8]
    var_10 = 3
    var_11 = var_1[var_10]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9]
    var_11 = var_4[var_9]
    var_12 = 2
    var_13 = var_4[var_12]
    var_14 = 3
    var_15 = var_4[var_14]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'AA99'
    var_2 = 'A'
    var_3 = '9'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[var_6]
    var_9 = 1
    var_10 = var_4[var_9]
    var_11 = var_4[var_9]
    var_12 = 2
    var_13 = var_4[var_12]
    var_14 = 3
    var_15 = var_4[var_14]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'same placeholder'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = ''
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    assert var_4 == ''

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'fixed'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    assert var_4 == 'fixed'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'A@1#'
    var_2 = '@'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = var_4[0]
    assert var_6 == 'A'
    var_7 = 1
    var_8 = var_4[var_7]
    var_9 = var_4[var_7]
    var_10 = var_4[2]
    assert var_10 == '1'
    var_11 = 3
    var_12 = var_4[var_11]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@'
    var_2 = 100
    var_3 = var_1 * var_2
    var_4 = '#'
    var_5 = var_4 * var_2
    var_6 = var_3 + var_5
    var_7 = var_0.generate_string_by_mask(var_6, var_1, var_4)
    var_8 = len(var_7)
    assert var_8 == 200
    var_9 = bool(var_2 and var_4)
    assert var_9 is True
    var_10 = bool(var_2)
    assert var_10 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 3/5 statements.
# Partially parsed test_constructor_without_seed_produces_different_instances. Retrieved 2/4 statements.
# Partially parsed test_constructor_initializes_random_state. Retrieved 1/3 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = type(var_0)
    var_2 = var_1.__name__
    assert var_2 == 'Random'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0.0



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 5/7 statements.
# Partially parsed test_constructor_with_none_seed. Retrieved 4/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)
    var_3 = 1
    var_4 = 100

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = 5
    var_3 = range(var_2)
    var_4 = 1
    var_5 = 100
    var_6 = [var_0.randint(var_4, var_5) for _ in var_3]
    var_7 = range(var_2)
    var_8 = [var_1.randint(var_4, var_5) for _ in var_7]
    var_9 = bool(var_6 != var_8)
    assert var_9 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)
    var_2 = 1
    var_3 = 10
    var_4 = 1



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_creates_instance_with_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_instance_with_int_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_instance_with_float_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_instance_with_str_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_instance_with_none_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_instance_with_bytes_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_creates_instance_with_bytearray_seed. Retrieved 3/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = bool(var_0 is not None)
    assert var_1 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = bytearray(var_0)
    var_2 = module_0.Random(var_1)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_int_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_none_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_float_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_str_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_bytes_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_bytearray_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_initializes_with_memoryview_seed. Retrieved 3/4 statements.
# Partially parsed test_constructor_initializes_with_version_argument. Retrieved 1/3 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 3.14
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 'seed'
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = bytearray(var_0)
    var_2 = module_0.Random(var_1)

import mimesis.random as module_0

def test_case_0():
    var_0 = b'seed'
    var_1 = memoryview(var_0)
    var_2 = module_0.Random(var_1)

def test_case_0():
    var_0 = 2



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_constructor_initializes_without_seed. Retrieved 1/2 statements.
# Partially parsed test_constructor_initializes_with_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_initializes_with_none. Retrieved 2/3 statements.
# Partially parsed test_constructor_instance_of_parent. Retrieved 1/3 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_same_seed_produces_same_sequence. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_different_seed_produces_different_sequence. Retrieved 4/6 statements.
# Partially parsed test_constructor_default_seed_produces_different_sequences. Retrieved 2/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = bool(var_0 is not var_1)
    assert var_2 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Random(var_0)
    var_2 = 2
    var_3 = module_0.Random(var_2)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_creates_instance. Retrieved 1/2 statements.
# Partially parsed test_constructor_accepts_seed. Retrieved 2/3 statements.
# Partially parsed test_constructor_with_same_seed_produces_same_random_sequence. Retrieved 3/5 statements.
# Partially parsed test_constructor_with_different_seed_produces_different_random_sequence. Retrieved 4/6 statements.
# Partially parsed test_constructor_default_instance_has_random_method. Retrieved 1/3 statements.
# Partially parsed test_constructor_instance_inherits_from_base_random. Retrieved 1/3 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()
    var_2 = bool(var_0 is not var_1)
    assert var_2 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = 1
    var_1 = module_0.Random(var_0)
    var_2 = 2
    var_3 = module_0.Random(var_2)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0.0

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



