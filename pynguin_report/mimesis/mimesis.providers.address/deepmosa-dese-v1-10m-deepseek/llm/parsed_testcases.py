####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_constructor_dataset_loaded. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = module_0.Address(var_0)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Address(seed=var_0)
    var_2 = module_0.Address(seed=var_0)
    var_3 = var_1.street_name()
    var_4 = var_2.street_name()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0._dataset
    var_2 = var_0._dataset
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.Address(var_0)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_format. Retrieved 3/4 statements.
# Partially parsed test_address_contains_street_number. Retrieved 2/4 statements.
# Partially parsed test_address_contains_street_name. Retrieved 5/8 statements.
# Partially parsed test_address_contains_street_suffix. Retrieved 5/8 statements.
# Partially parsed test_address_format_for_ja_locale. Retrieved 4/5 statements.
# Partially parsed test_address_contains_city_for_ja_locale. Retrieved 5/8 statements.
# Partially parsed test_address_contains_numbers_for_ja_locale. Retrieved 3/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'name'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'suffix'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = 'city'
    var_4 = [var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_address_constructor_default_locale. Retrieved 1/2 statements.
# Partially parsed test_address_constructor_custom_locale. Retrieved 2/3 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ru'
    var_1 = module_0.Address(var_0)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 123
    var_1 = module_0.Address(seed=var_0)
    var_2 = module_0.Address(seed=var_0)
    var_3 = var_1.street_number()
    var_4 = var_2.street_number()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = module_0.Address(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_shortened_locale. Retrieved 4/5 statements.
# Partially parsed test_address_japanese_locale. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = None
    var_1 = module_0.Address(seed=var_0)



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address_returns_shortened_format_when_locale_in_shortened_address_fmt. Retrieved 8/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = 'address_fmt'
    var_2 = [var_1]
    var_3 = '{st_num} {st_name}'
    var_4 = []
    var_5 = '123'
    var_6 = 'Main'
    var_7 = var_0.address()



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 10/15 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = module_0.Address(var_0)
    var_2 = [var_0]
    var_3 = '{st_num} {st_name}'
    var_4 = '123'
    var_5 = 'Main St'
    var_6 = 'address_fmt'
    var_7 = [var_6]
    var_8 = None
    var_9 = var_1.address()



# Parsed testcases at query #8
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()



# Parsed testcases at query #9
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = 123
    var_2 = module_0.Address(var_0, var_1)



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_shortened_format. Retrieved 4/7 statements.
# Partially parsed test_address_japanese_locale. Retrieved 4/7 statements.
# Partially parsed test_address_street_number_included. Retrieved 2/5 statements.
# Partially parsed test_address_street_name_included. Retrieved 5/8 statements.
# Partially parsed test_address_street_suffix_included. Retrieved 5/8 statements.
# Partially parsed test_address_consistent_format. Retrieved 8/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'name'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'suffix'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = 'address_fmt'
    var_2 = [var_1]
    var_3 = var_0.address()
    var_4 = var_0.street_number()
    var_5 = str(var_4)
    var_6 = var_0.street_name()
    var_7 = var_0.street_suffix()



# Parsed testcases at query #11
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_contains_street_number. Retrieved 2/4 statements.
# Partially parsed test_address_contains_street_name. Retrieved 2/4 statements.
# Partially parsed test_address_locale_specific_format. Retrieved 2/4 statements.
# Partially parsed test_address_shortened_format. Retrieved 2/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_constructor. Retrieved 2/4 statements.
# Partially parsed test_address_constructor_with_locale. Retrieved 3/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = module_0.Address(var_0)
    var_2 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = module_0.Address(seed=var_0)
    var_2 = module_0.Address(seed=var_0)
    var_3 = var_1.street_number()
    var_4 = var_2.street_number()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_returns_shortened_format_when_locale_in_shortened_address_fmt. Retrieved 8/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = 'address_fmt'
    var_2 = [var_1]
    var_3 = 'st_num={st_num}, st_name={st_name}'
    var_4 = []
    var_5 = '123'
    var_6 = 'Main'
    var_7 = var_0.address()
    assert var_7 == 'st_num=123, st_name=Main'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_address_locale_in_shortened_address_fmt. Retrieved 3/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_locale_default_is_not_missing_seed. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 12345



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + deepseek/deepseek-chat t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 8/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = 'address_fmt'
    var_2 = [var_1]
    var_3 = '{st_num} {st_name}'
    var_4 = []
    var_5 = '123'
    var_6 = 'Main'
    var_7 = var_0.address()
    assert var_7 == '123 Main'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_address_constructor_dataset_loaded. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = module_0.Address(seed=var_0)
    var_2 = module_0.Address(seed=var_0)
    var_3 = var_1.street_number()
    var_4 = var_2.street_number()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0._dataset
    var_2 = var_0._dataset
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = module_0.Address(var_0)



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address_constructor. Retrieved 3/6 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.locale
    var_2 = var_0._dataset



# Parsed testcases at query #5
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 9/13 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = module_0.Address(var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{st_num} {st_name}'
    var_5 = None
    var_6 = '123'
    var_7 = 'Main'
    var_8 = var_1.address()
    assert var_8 == '123 Main'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 8/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = 'address_fmt'
    var_2 = [var_1]
    var_3 = '{st_num} {st_name}'
    var_4 = []
    var_5 = '123'
    var_6 = 'Main'
    var_7 = var_0.address()



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_method_with_shortened_address_fmt. Retrieved 5/9 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = '{st_num} {st_name}'
    var_2 = '123'
    var_3 = 'Main St'
    var_4 = var_0.address()
    assert var_4 == '123 Main St'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address_returns_formatted_string. Retrieved 3/4 statements.
# Partially parsed test_address_contains_street_number. Retrieved 2/4 statements.
# Partially parsed test_address_contains_street_name. Retrieved 5/8 statements.
# Partially parsed test_address_contains_street_suffix_for_non_ja_locale. Retrieved 5/9 statements.
# Partially parsed test_address_japanese_format. Retrieved 4/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'name'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'suffix'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = var_0.street_number()
    var_3 = var_0.street_name()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'city'
    var_3 = [var_2]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_constructor. Retrieved 11/14 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = '_dataset'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._dataset
    var_4 = 'locale'
    var_5 = hasattr(var_0, var_4)
    var_6 = var_0.locale
    var_7 = 'fr'
    var_8 = module_0.Address(var_7)
    var_9 = 42
    var_10 = module_0.Address(seed=var_9)



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_address_returns_formatted_string. Retrieved 3/4 statements.
# Partially parsed test_address_contains_street_number. Retrieved 2/4 statements.
# Partially parsed test_address_contains_street_name. Retrieved 5/8 statements.
# Partially parsed test_address_ja_locale_has_different_format. Retrieved 6/9 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = 'street'
    var_3 = 'name'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = 0
    var_5 = result.split()[var_4]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()



# Parsed testcases at query #12
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 5/10 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = 'st_num: {st_num}, st_name: {st_name}'
    var_2 = '123'
    var_3 = 'Main St'
    var_4 = var_0.address()
    assert var_4 == 'st_num: 123, st_name: Main St'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 3/4 statements.
# Partially parsed test_address_ja_locale. Retrieved 4/5 statements.
# Partially parsed test_address_default_format. Retrieved 7/8 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = len(var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = var_1.street_number()
    var_5 = var_1.street_name()
    var_6 = var_1.street_suffix()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = var_0.street_number()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = var_0.street_name()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()
    var_2 = var_0.street_suffix()



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_address_constructor. Retrieved 8/13 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = '_dataset'
    var_2 = hasattr(var_0, var_1)
    var_3 = var_0._dataset
    var_4 = 'locale'
    var_5 = hasattr(var_0, var_4)
    var_6 = 12345
    var_7 = module_0.Address(seed=var_6)



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_predicate_at_line_12_evaluates_to_false. Retrieved 2/3 statements.


def test_case_0():
    var_0 = ()
    var_1 = {}



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_address_method_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_method_contains_street_number_and_name. Retrieved 2/6 statements.
# Partially parsed test_address_method_format_differs_by_locale. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()
    var_3 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = module_0.Address(var_0)
    var_2 = var_1.address()



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 8/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = 'en_GB'
    var_2 = [var_0, var_1]
    var_3 = module_0.Address(var_0)
    var_4 = '{st_num} {st_name}'
    var_5 = '123'
    var_6 = 'Main St'
    var_7 = var_3.address()
    assert var_7 == '123 Main St'



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_address_constructor. Retrieved 8/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = module_0.Address()
    var_1 = var_0._dataset
    var_2 = 'ru'
    var_3 = module_0.Address(var_2)
    var_4 = var_3._dataset
    var_5 = 12345
    var_6 = module_0.Address(seed=var_5)
    var_7 = var_6._dataset



