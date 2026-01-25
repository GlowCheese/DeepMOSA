####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_generate_string_by_mask_default. Retrieved 11/15 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 16/22 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 18/24 statements.
# Partially parsed test_generate_string_by_mask_literal_characters. Retrieved 14/19 statements.


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
    var_1 = 'A#B#C#'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 6
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6]
    var_8 = 2
    var_9 = var_2[var_8]
    var_10 = 3
    var_11 = var_2[var_10]
    var_12 = 4
    var_13 = var_2[var_12]
    var_14 = 5
    var_15 = var_2[var_14]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'X!Y!Z!'
    var_2 = 'X'
    var_3 = '!'
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

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'A#B#C#'
    var_2 = 'A'
    var_3 = var_0.generate_string_by_mask(var_1, var_2, var_2)
    var_4 = bool(False)
    assert var_4 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'A-#B-#C'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 7
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = var_2[1]
    assert var_6 == '-'
    var_7 = 2
    var_8 = var_2[var_7]
    var_9 = 3
    var_10 = var_2[var_9]
    var_11 = var_2[4]
    assert var_11 == '-'
    var_12 = 5
    var_13 = var_2[var_12]
    var_14 = 6
    var_15 = var_2[var_14]



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_generate_string_by_mask_default. Retrieved 11/15 statements.
# Partially parsed test_generate_string_by_mask_custom_mask. Retrieved 16/22 statements.
# Partially parsed test_generate_string_by_mask_custom_placeholders. Retrieved 18/24 statements.
# Partially parsed test_generate_string_by_mask_with_special_chars. Retrieved 18/24 statements.


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
    var_1 = 'A#B#C#'
    var_2 = var_0.generate_string_by_mask(var_1)
    var_3 = len(var_2)
    assert var_3 == 6
    var_4 = 0
    var_5 = var_2[var_4]
    var_6 = 1
    var_7 = var_2[var_6]
    var_8 = 2
    var_9 = var_2[var_8]
    var_10 = 3
    var_11 = var_2[var_10]
    var_12 = 4
    var_13 = var_2[var_12]
    var_14 = 5
    var_15 = var_2[var_14]

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'X@Y@Z@'
    var_2 = '@'
    var_3 = 'X'
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

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = '#'
    var_2 = var_0.generate_string_by_mask(char=var_1, digit=var_1)
    var_3 = bool(False)
    assert var_3 is True

import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()
    var_1 = 'A-#B-#C-#'
    var_2 = 'A'
    var_3 = '#'
    var_4 = var_0.generate_string_by_mask(var_1, var_2, var_3)
    var_5 = len(var_4)
    assert var_5 == 9
    var_6 = 0
    var_7 = var_4[var_6]
    var_8 = var_4[1]
    assert var_8 == '-'
    var_9 = 2
    var_10 = var_4[var_9]
    var_11 = 3
    var_12 = var_4[var_11]
    var_13 = var_4[4]
    assert var_13 == '-'
    var_14 = 5
    var_15 = var_4[var_14]
    var_16 = 6
    var_17 = var_4[var_16]
    var_18 = var_4[7]
    assert var_18 == '-'
    var_19 = 8
    var_20 = var_4[var_19]



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_random_constructor_initialization. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_random_constructor. Retrieved 1/4 statements.


import mimesis.random as module_0

def test_case_0():
    var_0 = module_0.Random()



