####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_string_by_mask_default_mask. Retrieved 12/18 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 13/19 statements.
# Partially parsed test_generate_string_by_mask_with_static_chars. Retrieved 9/13 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 15/21 statements.
# Partially parsed test_generate_string_by_mask_long_mask. Retrieved 10/18 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = var_1.generate_string_by_mask()
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6]
    var_8 = 2
    var_9 = var_2[var_8]
    var_10 = 3
    var_11 = var_2[var_10]

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = '@@##'
    var_3 = var_1.generate_string_by_mask(var_2)
    var_4 = len(var_3)
    assert var_4 == 4
    var_5 = 0
    var_6 = var_3[var_5]
    var_7 = 1
    var_8 = var_3[var_7]
    var_9 = 2
    var_10 = var_3[var_9]
    var_11 = 3
    var_12 = var_3[var_11]

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = 'ABC@#XYZ'
    var_3 = var_1.generate_string_by_mask(var_2)
    var_4 = len(var_3)
    assert var_4 == 8
    var_5 = var_3[0]
    assert var_5 == 'A'
    var_6 = var_3[1]
    assert var_6 == 'B'
    var_7 = var_3[2]
    assert var_7 == 'C'
    var_8 = 3
    var_9 = var_3[var_8]
    var_10 = 4
    var_11 = var_3[var_10]
    var_12 = var_3[5]
    assert var_12 == 'X'
    var_13 = var_3[6]
    assert var_13 == 'Y'
    var_14 = var_3[7]
    assert var_14 == 'Z'

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = 'xxyy'
    var_3 = 'x'
    var_4 = 'y'
    var_5 = var_1.generate_string_by_mask(var_2, var_3, var_4)
    var_6 = len(var_5)
    assert var_6 == 4
    var_7 = 0
    var_8 = var_5[var_7]
    var_9 = 1
    var_10 = var_5[var_9]
    var_11 = 2
    var_12 = var_5[var_11]
    var_13 = 3
    var_14 = var_5[var_13]

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = '@'
    var_3 = var_1.generate_string_by_mask(char=var_2, digit=var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'same placeholder'
    var_6 = bool('same placeholder' in str(e).lower())
    assert var_6 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = ''
    var_3 = var_1.generate_string_by_mask(var_2)
    assert var_3 == ''

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = 'HELLO'
    var_3 = var_1.generate_string_by_mask(var_2)
    assert var_3 == 'HELLO'

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = '@'
    var_3 = 10
    var_4 = var_2 * var_3
    var_5 = '#'
    var_6 = var_5 * var_3
    var_7 = var_4 + var_6
    var_8 = var_1.generate_string_by_mask(var_7)
    var_9 = len(var_8)
    assert var_9 == 20
    var_10 = bool(var_2)
    assert var_10 is True
    var_11 = bool(var_2)
    assert var_11 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor. Retrieved 23/25 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True
    var_22 = 'random'
    var_23 = hasattr(var_0, var_22)
    var_24 = bool(var_23)
    assert var_24 is True
    var_25 = 'choices'
    var_26 = hasattr(var_0, var_25)
    var_27 = bool(var_26)
    assert var_27 is True
    var_28 = 'choice'
    var_29 = hasattr(var_0, var_28)
    var_30 = bool(var_29)
    assert var_30 is True
    var_31 = 'getrandbits'
    var_32 = hasattr(var_0, var_31)
    var_33 = bool(var_32)
    assert var_33 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_random_constructor. Retrieved 16/18 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test that Random class can be instantiated.'
    var_1 = module_0.Random()
    var_2 = 'randints'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = '_generate_string'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = 'generate_string_by_mask'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True
    var_11 = 'uniform'
    var_12 = hasattr(var_1, var_11)
    var_13 = bool(var_12)
    assert var_13 is True
    var_14 = 'randbytes'
    var_15 = hasattr(var_1, var_14)
    var_16 = bool(var_15)
    assert var_16 is True
    var_17 = 'weighted_choice'
    var_18 = hasattr(var_1, var_17)
    var_19 = bool(var_18)
    assert var_19 is True
    var_20 = 'choice_enum_item'
    var_21 = hasattr(var_1, var_20)
    var_22 = bool(var_21)
    assert var_22 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_string_by_mask_default_mask. Retrieved 11/16 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 12/17 statements.
# Partially parsed test_generate_string_by_mask_all_chars. Retrieved 4/7 statements.
# Partially parsed test_generate_string_by_mask_all_digits. Retrieved 4/7 statements.
# Partially parsed test_generate_string_by_mask_with_static_chars. Retrieved 10/14 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 14/19 statements.
# Partially parsed test_generate_string_by_mask_long_mask. Retrieved 4/11 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5]
    var_7 = 2
    var_8 = var_1[var_7]
    var_9 = 3
    var_10 = var_1[var_9]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6]
    var_8 = 2
    var_9 = var_2[var_8]
    var_10 = 3
    var_11 = var_2[var_10]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@@'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '####'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@-#-@'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 5
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = var_2[1]
    assert var_6 == '-'
    var_7 = 2
    var_8 = var_2[var_7]
    var_9 = var_2[3]
    assert var_9 == '-'
    var_10 = 4
    var_11 = var_2[var_10]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'X*X*'
    var_2 = 'X'
    var_3 = '*'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]
    var_12 = 3
    var_13 = var_4[var_12]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = ''
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == ''

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'ABC-XYZ'
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == 'ABC-XYZ'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@#@#'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'same placeholder'
    var_6 = bool('same placeholder' in str(e).lower())
    assert var_6 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@#@#@#@#@#'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 10
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = bool(var_3)
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor. Retrieved 2/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 'Test that Random class can be instantiated.'
    var_1 = module_0.Random()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/18 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = 'randints'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_random_constructor. Retrieved 15/17 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'randints'
    var_2 = hasattr(var_0, var_1)
    var_3 = bool(var_2)
    assert var_3 is True
    var_4 = '_generate_string'
    var_5 = hasattr(var_0, var_4)
    var_6 = bool(var_5)
    assert var_6 is True
    var_7 = 'generate_string_by_mask'
    var_8 = hasattr(var_0, var_7)
    var_9 = bool(var_8)
    assert var_9 is True
    var_10 = 'uniform'
    var_11 = hasattr(var_0, var_10)
    var_12 = bool(var_11)
    assert var_12 is True
    var_13 = 'randbytes'
    var_14 = hasattr(var_0, var_13)
    var_15 = bool(var_14)
    assert var_15 is True
    var_16 = 'weighted_choice'
    var_17 = hasattr(var_0, var_16)
    var_18 = bool(var_17)
    assert var_18 is True
    var_19 = 'choice_enum_item'
    var_20 = hasattr(var_0, var_19)
    var_21 = bool(var_20)
    assert var_21 is True



