####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------






# Parsed testcases at query #2
#--------------------------






# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------






# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------






# Parsed testcases at query #7
#--------------------------






# Parsed testcases at query #8
#--------------------------






# Parsed testcases at query #9
#--------------------------






# Parsed testcases at query #10
#--------------------------






# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------






# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------






# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 10/13 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #20
#--------------------------






# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------






# Parsed testcases at query #24
#--------------------------






# Parsed testcases at query #25
#--------------------------






# Parsed testcases at query #26
#--------------------------






# Parsed testcases at query #27
#--------------------------






# Parsed testcases at query #28
#--------------------------






# Parsed testcases at query #29
#--------------------------






# Parsed testcases at query #30
#--------------------------






# Parsed testcases at query #31
#--------------------------






# Parsed testcases at query #32
#--------------------------






####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_nationality_with_gender. Retrieved 1/7 statements.
# Partially parsed test_nationality_without_gender. Retrieved 2/3 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'INVALID_GENDER'
    var_2 = var_0.nationality(var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_nationality_with_gender_separated. Retrieved 8/11 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = 'American'
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = {var_1: var_5, var_2: var_6}



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_username_default_mask. Retrieved 3/8 statements.
# Partially parsed test_username_custom_mask. Retrieved 12/15 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.username()
    var_2 = len(var_1)

import mimesis.providers.person as module_0
import re as module_1

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'C_C_d'
    var_2 = var_0.username(var_1)
    var_3 = '_'
    var_4 = module_1.split(var_3)
    var_5 = len(var_4)
    assert var_5 == 3
    var_6 = 0
    var_7 = var_4[var_6][var_6]
    var_8 = 1
    var_9 = var_4[var_8][var_6]
    var_10 = 2
    var_11 = var_4[var_10]

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 2000
    var_2 = 2020
    var_3 = (var_1, var_2)
    var_4 = var_0.username(drange=var_3)
    var_5 = -1
    var_6 = '_'
    var_7 = username.split(var_6)[var_5]
    var_8 = int(var_7)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 2000
    var_2 = (var_1,)
    var_3 = var_0.username(drange=var_2)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'invalid'
    var_2 = var_0.username(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'd.d.d'
    var_2 = var_0.username(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_patronymic_with_valid_gender. Retrieved 1/3 statements.
# Partially parsed test_patronymic_with_non_ru_uk_locale. Retrieved 2/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'INVALID_GENDER'
    var_2 = var_0.patronymic(var_1)
    assert var_2 is None

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Person()



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    assert var_1 is None



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics_available. Retrieved 2/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Person()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_surname_returns_string. Retrieved 2/3 statements.
# Partially parsed test_surname_with_gender_returns_string. Retrieved 1/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'invalid'
    var_2 = var_0.surname(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = var_0.surname()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Person()
    var_2 = var_1.surname()
    var_3 = module_0.Person()
    var_4 = var_3.surname()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 10/13 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_nationality_with_gender_dict.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 10/13 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics_available. Retrieved 2/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Person()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_surname_with_gender_separated_dict. Retrieved 10/13 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_nationality_returns_string. Retrieved 2/3 statements.
# Partially parsed test_nationality_with_gender_returns_string. Retrieved 1/7 statements.
# Partially parsed test_nationality_values_are_from_data. Retrieved 6/9 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'invalid'
    var_2 = var_0.nationality(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'nationality'
    var_2 = [var_1]
    var_3 = list(var_1)
    var_4 = [item for sublist in var_3 for item in sublist]
    var_5 = var_0.nationality()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics_available. Retrieved 2/5 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = []



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 8/11 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = [var_3]
    var_5 = 'Johnson'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_nationality_returns_string_when_nationalities_is_dict. Retrieved 8/12 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'American'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_surname_returns_string. Retrieved 2/3 statements.
# Partially parsed test_surname_with_gender_returns_string. Retrieved 1/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'invalid'
    var_2 = var_0.surname(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = var_0.surname()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Person()
    var_2 = var_1.surname()
    var_3 = var_1.surname()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_patronymic_returns_string_for_supported_locale. Retrieved 3/4 statements.
# Partially parsed test_patronymic_returns_gender_specific_result. Retrieved 2/8 statements.
# Partially parsed test_patronymic_returns_random_results. Retrieved 4/7 statements.
# Partially parsed test_patronymic_handles_none_gender. Retrieved 4/5 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Person()
    var_2 = var_1.patronymic()
    assert var_2 is None

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Person()
    var_2 = var_1.patronymic()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Person()
    var_2 = 10
    var_3 = range(var_2)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Person()
    var_2 = None
    var_3 = var_1.patronymic(var_2)



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_nationality_with_gender_specific_data. Retrieved 8/11 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'French'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 8/11 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'American'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_surname_method_generates_random_surname. Retrieved 3/4 statements.
# Partially parsed test_surname_method_generates_surname_for_specified_gender. Retrieved 1/9 statements.
# Partially parsed test_surname_method_generates_surname_for_default_gender. Retrieved 3/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()
    var_2 = len(var_1)



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics_available. Retrieved 2/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Person()



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics_available. Retrieved 2/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Person()



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_surname_returns_string. Retrieved 2/3 statements.
# Partially parsed test_surname_with_gender_returns_string. Retrieved 1/4 statements.
# Partially parsed test_surname_with_gender_returns_non_empty_string. Retrieved 1/3 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'INVALID'
    var_2 = var_0.surname(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.surname()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_nationality_with_gender_specific_data. Retrieved 8/11 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'French'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}



