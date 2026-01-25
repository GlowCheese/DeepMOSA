####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_string_by_mask_with_default_parameters. Retrieved 7/9 statements.
# Partially parsed test_generate_string_by_mask_with_custom_mask. Retrieved 11/14 statements.
# Partially parsed test_generate_string_by_mask_with_custom_placeholders. Retrieved 11/13 statements.
# Partially parsed test_generate_string_by_mask_with_mixed_placeholders. Retrieved 18/24 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = var_0.generate_string_by_mask()
    var_2 = len(var_1)
    assert var_2 == 4
    var_3 = 0
    var_4 = var_1[var_3]
    var_5 = 1
    var_6 = var_1[var_5:]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##@@'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 6
    var_4 = 0
    var_5 = 2
    var_6 = var_2[var_4:var_5]
    var_7 = 4
    var_8 = var_2[var_5:var_7]
    var_9 = 6
    var_10 = var_2[var_7:var_9]

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
    var_9 = 4
    var_10 = var_4[var_7:var_9]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@@'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'ABC-123'
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == 'ABC-123'
    var_3 = len(var_2)
    assert var_3 == 7

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'A1B2C3'
    var_2 = 'A'
    var_3 = '1'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 6
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = 1
    var_9 = var_4[var_8]
    var_10 = 2
    var_11 = var_4[var_10]
    var_12 = 3
    var_13 = var_4[var_12]
    var_14 = 4
    var_15 = var_4[var_14]
    var_16 = 5
    var_17 = var_4[var_16]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_constructor_initializes_random_instance. Retrieved 5/6 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.random
    var_4 = callable(var_3)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_constructor_initializes_random_instance. Retrieved 5/6 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.random
    var_4 = callable(var_3)



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_random_constructor_default. Retrieved 1/2 statements.
# Partially parsed test_random_constructor_with_seed. Retrieved 2/3 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_constructor_initialization. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_string_by_mask_basic. Retrieved 8/11 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 8/11 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 10/13 statements.
# Partially parsed test_generate_string_by_mask_long_mask. Retrieved 18/22 statements.


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
    var_7 = var_1[var_6:]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 2
    var_5 = var_2[:var_4]
    var_6 = var_2[:var_4]
    var_7 = var_2[var_4:]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'AA99'
    var_2 = 'A'
    var_3 = '9'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 4
    var_6 = 2
    var_7 = var_4[:var_6]
    var_8 = var_4[:var_6]
    var_9 = var_4[var_6:]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@@@'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'A1B2'
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == 'A1B2'

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = ''
    var_2 = var_0.generate_string_by_mask(var_1)
    assert var_2 == ''

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@@##@@##@@##'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 12
    var_4 = 0
    var_5 = 1
    var_6 = 4
    var_7 = 5
    var_8 = 8
    var_9 = 9
    var_10 = [var_4, var_5, var_6, var_7, var_8, var_9]
    var_11 = 2
    var_12 = 3
    var_13 = 6
    var_14 = 7
    var_15 = 10
    var_16 = 11
    var_17 = [var_11, var_12, var_13, var_14, var_15, var_16]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor. Retrieved 5/6 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'random'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0.random
    var_4 = callable(var_3)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.
# Partially parsed test_randints_positive. Retrieved 6/8 statements.
# Partially parsed test_generate_string. Retrieved 3/7 statements.
# Partially parsed test_generate_string_by_mask. Retrieved 8/10 statements.
# Partially parsed test_randbytes. Retrieved 4/5 statements.
# Partially parsed test_choice_enum_item. Retrieved 3/5 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 5
    var_2 = 1
    var_3 = 10
    var_4 = var_0.randints(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 0
    var_2 = 1
    var_3 = 10
    var_4 = var_0.randints(var_1, var_2, var_3)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'abc'
    var_2 = 5

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 4
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6:]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '@###'
    var_2 = '@'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1.0
    var_2 = 2.0
    var_3 = var_0.uniform(var_1, var_2)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 10
    var_2 = var_0.randbytes(var_1)
    var_3 = len(var_2)
    assert var_3 == 10

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'a'
    var_2 = 'b'
    var_3 = 0.5
    var_4 = {var_1: var_3, var_2: var_3}
    var_5 = var_0.weighted_choice(var_4)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = {}
    var_2 = var_0.weighted_choice(var_1)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 1
    var_2 = 2



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_Random_constructor_with_seed. Retrieved 3/5 statements.
# Partially parsed test_Random_constructor_without_seed. Retrieved 2/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Random(var_0)
    var_2 = module_0.Random(var_0)

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = module_0.Random()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/2 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



