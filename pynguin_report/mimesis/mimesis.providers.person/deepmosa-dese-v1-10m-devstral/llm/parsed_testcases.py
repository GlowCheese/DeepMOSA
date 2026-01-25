####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_surname_default. Retrieved 2/3 statements.
# Partially parsed test_surname_with_gender. Retrieved 1/7 statements.
# Partially parsed test_surname_with_none_gender. Retrieved 3/4 statements.


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
    var_1 = None
    var_2 = var_0.surname(var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_surname_with_gender_dict. Retrieved 9/15 statements.


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
    var_8 = 0



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_patronymic_with_valid_gender. Retrieved 1/4 statements.
# Partially parsed test_patronymic_with_no_gender. Retrieved 2/3 statements.
# Partially parsed test_patronymic_with_none_gender. Retrieved 3/4 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'invalid_gender'
    var_2 = var_0.patronymic(var_1)
    assert var_2 is None

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = None
    var_2 = var_0.patronymic(var_1)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_email_default. Retrieved 5/6 statements.
# Partially parsed test_email_custom_domains. Retrieved 7/10 statements.
# Partially parsed test_email_unique. Retrieved 7/8 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.email()
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'example.com'
    var_2 = 'test.org'
    var_3 = [var_1, var_2]
    var_4 = var_0.email(var_3)
    var_5 = '@example.com'
    var_6 = '@test.org'

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = True
    var_2 = var_0.email(unique=var_1)
    var_3 = 0
    var_4 = '@'
    var_5 = email.split(var_4)[var_3]
    var_6 = len(var_5)
    assert var_6 == 32

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Person()
    var_2 = True
    var_3 = var_1.email(unique=var_2)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_patronymic_with_gender. Retrieved 1/4 statements.
# Partially parsed test_patronymic_without_gender. Retrieved 2/3 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = module_0.Person()
    var_2 = var_1.patronymic()
    assert var_2 is None



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = 42
    var_2 = module_0.Person()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics_available. Retrieved 2/3 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    assert var_1 is None



# Parsed testcases at query #8
#--------------------------




import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Person()
    var_2 = True
    var_3 = var_1.email(unique=var_2)



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_nationality_without_gender. Retrieved 3/4 statements.
# Partially parsed test_nationality_with_gender. Retrieved 1/5 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.nationality()
    var_2 = len(var_1)

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()

import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = 'invalid_gender'
    var_2 = var_0.nationality(var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_nationality_with_gender_separated_data. Retrieved 11/17 statements.


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
    var_8 = 0
    var_9 = 'nationality'
    var_10 = [var_9]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics. Retrieved 2/3 statements.


import mimesis.providers.person as module_0

def test_case_0():
    var_0 = module_0.Person()
    var_1 = var_0.patronymic()
    assert var_1 is None



# Parsed testcases at query #12
#--------------------------




import mimesis.providers.person as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Person()
    var_2 = True
    var_3 = var_1.email(unique=var_2)



