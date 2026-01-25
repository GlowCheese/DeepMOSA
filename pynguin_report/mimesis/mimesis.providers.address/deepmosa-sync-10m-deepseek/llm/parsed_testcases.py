####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_shortened_locale. Retrieved 4/5 statements.
# Partially parsed test_address_japanese_locale. Retrieved 4/5 statements.
# Partially parsed test_address_street_number_included. Retrieved 2/4 statements.
# Partially parsed test_address_street_name_included. Retrieved 5/8 statements.
# Partially parsed test_address_street_suffix_included_for_non_ja. Retrieved 6/9 statements.
# Partially parsed test_address_format_follows_locale_pattern. Retrieved 6/9 statements.
# Partially parsed test_address_ja_locale_uses_city. Retrieved 5/8 statements.
# Partially parsed test_address_ja_locale_contains_numbers. Retrieved 3/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 'street'
    var_4 = 'name'
    var_5 = [var_3, var_4]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'street'
    var_5 = 'suffix'
    var_6 = [var_4, var_5]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = var_2.address()
    var_6 = ' '

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'city'
    var_5 = [var_4]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = bool(var_2 != '')
    assert var_3 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_enum.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = 123
    var_2 = 'test'
    var_3 = 'extra_arg'
    var_4 = {var_3: var_2}
    var_5 = module_0.Address(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 == 'de'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'fr'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = var_2._dataset
    var_7 = bool(var_2._dataset != var_5._dataset)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-US'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(var_2._dataset)
    assert var_3 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = var_1.random
    var_6 = bool(var_1.random is not None)
    assert var_6 is True



# Parsed testcases at query #3
#--------------------------






# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 11/18 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '{st_num} {st_name}'
    var_3 = '123'
    var_4 = 'Main'
    var_5 = 'St'
    var_6 = 'en_US'
    var_7 = 'en_GB'
    var_8 = [var_6, var_7]
    var_9 = var_1.address()
    assert var_9 == '123 Main'
    var_10 = '{st_num} {st_name} {st_sfx}'
    var_11 = var_1.address()
    assert var_11 == '123 Main St'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_contains_street_number_and_name_for_shortened_locale. Retrieved 4/5 statements.
# Partially parsed test_address_uses_shortened_format_for_supported_locale. Retrieved 7/9 statements.
# Partially parsed test_address_uses_japanese_format_for_ja_locale. Retrieved 12/16 statements.
# Partially parsed test_address_includes_street_suffix_for_default_locale. Retrieved 8/10 statements.
# Partially parsed test_address_calls_street_number_and_street_name. Retrieved 10/14 statements.
# Partially parsed test_address_handles_random_choice_for_street_name. Retrieved 9/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = var_1.street_number()
    var_4 = bool(var_3 in var_2)
    assert var_4 is True
    var_5 = var_1.street_name()
    var_6 = bool(var_5 in var_2)
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = 'st_num st_name'
    var_5 = []
    var_6 = var_1.address()
    var_7 = f'{var_1.street_number()} {var_1.street_name()}'
    var_8 = bool(var_6 == var_7)
    assert var_8 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{} {}-{}-{}'
    var_5 = 'Tokyo'
    var_6 = [var_5]
    var_7 = 0
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = var_1.address()
    assert var_12 == 'Tokyo 1-2-3'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = 'st_num st_name st_sfx'
    var_5 = 'Ave'
    var_6 = [var_5]
    var_7 = var_1.address()
    var_8 = var_1.street_suffix()
    var_9 = bool(var_8 in var_7)
    assert var_9 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '123'
    var_3 = 'Main'
    var_4 = 'St'
    var_5 = 'address_fmt'
    var_6 = [var_5]
    var_7 = 'st_num st_name st_sfx'
    var_8 = 'Ave'
    var_9 = [var_8]
    var_10 = var_1.address()
    assert var_10 == '123 Main St'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'Elm Street'
    var_3 = 'street'
    var_4 = 'name'
    var_5 = [var_3, var_4]
    var_6 = 'Oak Avenue'
    var_7 = [var_2, var_6]
    var_8 = 'st_num st_name st_sfx'
    var_9 = var_1.address()
    var_10 = 'Elm Street'
    var_11 = bool('Elm Street' in var_9)
    assert var_11 is True



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 7/10 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Address(var_0, **var_2)
    var_4 = '{st_num} {st_name}'
    var_5 = '123'
    var_6 = 'Main'
    var_7 = var_3.address()
    assert var_7 == '123 Main'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_locale_and_seed. Retrieved 4/5 statements.
# Partially parsed test_address_initialization_without_datafile. Retrieved 2/5 statements.
# Partially parsed test_address_initialization_with_locale_separator. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'
    var_5 = var_3._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'fr'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = var_2.city()
    var_7 = var_5.city()
    var_8 = bool(var_6 != var_7)
    assert var_8 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset != {})
    assert var_3 is True

def test_case_0():
    var_0 = 'custom'
    var_1 = ''

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <it>'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en-gb'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-gb'
    var_4 = var_2._dataset



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_enum.
# Partially parsed test_address_update_dataset. Retrieved 8/13 statements.
# Partially parsed test_address_get_current_locale. Retrieved 2/3 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'UnsupportedLocale'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <de>'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'city'
    var_3 = [var_2]
    var_4 = 'NewCity1'
    var_5 = 'NewCity2'
    var_6 = [var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = [var_2]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_extract'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = '_load_dataset'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = 'update_dataset'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address_constructor_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_constructor_custom_locale. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True
    var_8 = var_2.street_name()
    var_9 = var_4.street_name()
    var_10 = bool(var_8 == var_9)
    assert var_10 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'UnsupportedLocale'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'street'
    var_3 = bool('street' in var_1._dataset)
    assert var_3 is True
    var_4 = 'name'
    var_5 = bool('name' in var_1._dataset['street'])
    assert var_5 is True
    var_6 = 'suffix'
    var_7 = bool('suffix' in var_1._dataset['street'])
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'fr'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = var_2._dataset
    var_7 = bool(var_2._dataset != var_5._dataset)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42
    var_4 = 'random'
    var_5 = hasattr(var_2, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <it>'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_contains_street_number_and_name. Retrieved 2/6 statements.
# Partially parsed test_address_for_shortened_locale. Retrieved 2/6 statements.
# Partially parsed test_address_for_ja_locale. Retrieved 2/6 statements.
# Partially parsed test_address_for_regular_locale. Retrieved 2/6 statements.
# Partially parsed test_address_street_name_from_list. Retrieved 2/3 statements.
# Partially parsed test_address_street_suffix_from_list. Retrieved 2/3 statements.
# Partially parsed test_address_format_uses_street_suffix_for_non_special_locales. Retrieved 3/6 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 100
    var_3 = var_1.street_number(var_2)
    var_4 = int(var_3)
    var_5 = 1
    var_6 = bool(1 <= var_4)
    assert var_6 is True
    var_7 = bool(var_4 <= 100)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_name()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_suffix()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_number()
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_name()
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_suffix()
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_address_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_shortened_locale. Retrieved 4/5 statements.
# Partially parsed test_address_japanese_locale. Retrieved 4/5 statements.
# Partially parsed test_address_street_number_included. Retrieved 2/4 statements.
# Partially parsed test_address_street_name_included. Retrieved 5/8 statements.
# Partially parsed test_address_street_suffix_included_for_non_ja. Retrieved 6/9 statements.
# Partially parsed test_address_format_follows_locale_pattern. Retrieved 7/10 statements.
# Partially parsed test_address_ja_format_contains_city. Retrieved 5/8 statements.
# Partially parsed test_address_ja_format_contains_numbers. Retrieved 3/5 statements.
# Partially parsed test_address_shortened_format_excludes_suffix. Retrieved 6/9 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 'street'
    var_4 = 'name'
    var_5 = [var_3, var_4]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'street'
    var_5 = 'suffix'
    var_6 = [var_4, var_5]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_GB'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = var_2.address()
    var_6 = '{'
    var_7 = '}'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'city'
    var_5 = [var_4]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'street'
    var_4 = 'suffix'
    var_5 = [var_3, var_4]
    var_6 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.address()
    var_6 = var_4.address()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 10
    var_3 = range(var_2)
    var_4 = [var_1.address() for _ in var_3]
    var_5 = set(var_4)
    var_6 = len(var_5)
    var_7 = bool(var_6 > 1)
    assert var_7 is True



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_locale_in_shortened_address_fmt. Retrieved 8/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = 'en_GB'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Address(var_0, **var_3)
    var_5 = '{st_num} {st_name}'
    var_6 = '123'
    var_7 = 'Main'
    var_8 = var_4.address()
    assert var_8 == '123 Main'



# Parsed testcases at query #14
#--------------------------






# Parsed testcases at query #15
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_correctly_for_shortened_locale. Retrieved 10/14 statements.
# Partially parsed test_address_formats_correctly_for_ja_locale. Retrieved 14/18 statements.
# Partially parsed test_address_formats_correctly_for_default_locale. Retrieved 12/17 statements.
# Partially parsed test_address_uses_street_number_and_name. Retrieved 11/15 statements.
# Partially parsed test_address_includes_street_suffix_for_non_shortened_locale. Retrieved 12/17 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = '{st_num} {st_name}'
    var_6 = [var_5]
    var_7 = []
    var_8 = '123'
    var_9 = 'Main'
    var_10 = var_2.address()
    assert var_10 == '123 Main'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = '{}{}{}{}'
    var_6 = [var_5]
    var_7 = 'Tokyo'
    var_8 = [var_7]
    var_9 = 0
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = [var_10, var_11, var_12]
    var_14 = var_2.address()
    assert var_14 == 'Tokyo123'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = '{st_num} {st_name} {st_sfx}'
    var_6 = [var_5]
    var_7 = 'St'
    var_8 = [var_7]
    var_9 = '456'
    var_10 = 'Oak'
    var_11 = 'Ave'
    var_12 = var_2.address()
    assert var_12 == '456 Oak Ave'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{st_num} {st_name} {st_sfx}'
    var_5 = [var_4]
    var_6 = 'St'
    var_7 = [var_6]
    var_8 = '789'
    var_9 = 'Pine'
    var_10 = 'Rd'
    var_11 = var_1.address()
    var_12 = '789'
    var_13 = bool('789' in var_11)
    assert var_13 is True
    var_14 = 'Pine'
    var_15 = bool('Pine' in var_11)
    assert var_15 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = '{st_num} {st_name} {st_sfx}'
    var_6 = [var_5]
    var_7 = 'St'
    var_8 = [var_7]
    var_9 = '101'
    var_10 = 'Elm'
    var_11 = 'Blvd'
    var_12 = var_2.address()
    assert var_12 == '101 Elm Blvd'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_object.
# Partially parsed test_address_initialization_with_additional_arguments. Retrieved 5/6 statements.
# Partially parsed test_address_initialization_with_locale_separator. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_dataset_structure. Retrieved 4/7 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = 42
    var_2 = 'test'
    var_3 = 'extra_arg'
    var_4 = {var_3: var_2}
    var_5 = module_0.Address(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 == 'de'
    var_7 = var_5._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'fr'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = var_2.city()
    var_7 = var_5.city()
    var_8 = bool(var_6 != var_7)
    assert var_8 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en-gb'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-gb'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'street'
    var_3 = 'name'
    var_4 = [var_2, var_3]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = {}
    var_3 = module_0.Address(**var_2)
    var_4 = var_1.street_number()
    var_5 = var_3.street_number()
    var_6 = bool(var_4 != var_5)
    assert var_6 is True



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_correctly_for_shortened_locale. Retrieved 9/12 statements.
# Partially parsed test_address_formats_correctly_for_ja_locale. Retrieved 13/16 statements.
# Partially parsed test_address_formats_correctly_for_default_locale. Retrieved 18/22 statements.
# Partially parsed test_address_uses_street_number_and_name. Retrieved 8/11 statements.
# Partially parsed test_address_handles_empty_street_name. Retrieved 8/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = 'Short {st_num} {st_name}'
    var_6 = []
    var_7 = '123'
    var_8 = 'Main'
    var_9 = var_2.address()
    assert var_9 == 'Short 123 Main'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'city'
    var_4 = [var_3]
    var_5 = 'Tokyo'
    var_6 = [var_5]
    var_7 = '{0}{1}{2}{3}'
    var_8 = 0
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = var_2.address()
    assert var_13 == 'Tokyo123'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'street'
    var_4 = 'suffix'
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = [var_3, var_4]
    var_8 = 'Ave'
    var_9 = [var_8]
    var_10 = 'name'
    var_11 = [var_3, var_10]
    var_12 = 'Main'
    var_13 = [var_12]
    var_14 = '{st_num} {st_name} {st_sfx}'
    var_15 = '456'
    var_16 = 'Oak'
    var_17 = 'St'
    var_18 = var_2.address()
    assert var_18 == '456 Oak St'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{st_num} {st_name}'
    var_5 = []
    var_6 = '789'
    var_7 = 'Pine'
    var_8 = var_1.address()
    assert var_8 == '789 Pine'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{st_num} {st_name}'
    var_5 = ''
    var_6 = [var_5]
    var_7 = '999'
    var_8 = var_1.address()
    assert var_8 == '999 '



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 25/35 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Address(var_0, **var_2)
    var_4 = 'address_fmt'
    var_5 = [var_4]
    var_6 = '{st_num} {st_name}'
    var_7 = []
    var_8 = '123'
    var_9 = 'Main'
    var_10 = var_3.address()
    assert var_10 == '123 Main'
    var_11 = [var_4]
    var_12 = '{st_num} {st_name} {st_sfx}'
    var_13 = []
    var_14 = 'St'
    var_15 = var_3.address()
    assert var_15 == '123 Main St'
    var_16 = [var_4]
    var_17 = '{} {}-{}-{}'
    var_18 = 'Tokyo'
    var_19 = [var_18]
    var_20 = 0
    var_21 = 1
    var_22 = 2
    var_23 = 3
    var_24 = [var_21, var_22, var_23]
    var_25 = var_3.address()
    assert var_25 == 'Tokyo 1-2-3'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 7/10 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Address(var_0, **var_2)
    var_4 = '{st_num} {st_name}'
    var_5 = '123'
    var_6 = 'Main'
    var_7 = var_3.address()
    assert var_7 == '123 Main'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_locale_and_seed. Retrieved 4/5 statements.
# Failed to parse test_address_initialization_with_locale_object.
# Partially parsed test_address_initialization_without_locale_dependent_data. Retrieved 2/5 statements.
# Partially parsed test_address_initialization_with_master_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_invalid_seed_type. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_additional_args. Retrieved 4/7 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'
    var_5 = var_3._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'UnsupportedLocale'

def test_case_0():
    var_0 = 'custom_address'
    var_1 = ''

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-US'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid_seed'
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2._dataset

def test_case_0():
    var_0 = 'en'
    var_1 = 12345
    var_2 = 'extra_arg'
    var_3 = 'value'
    var_4 = [var_2]



# Parsed testcases at query #5
#--------------------------






# Parsed testcases at query #6
#--------------------------

# Failed to parse test_locale_setup_before_dataset_load.




# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_enum.
# Partially parsed test_address_get_current_locale. Retrieved 2/3 statements.
# Partially parsed test_address_update_dataset. Retrieved 8/13 statements.
# Partially parsed test_address_update_dataset_with_invalid_data_raises_error. Retrieved 2/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'UnsupportedLocale'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <de>'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'city'
    var_3 = [var_2]
    var_4 = 'NewCity1'
    var_5 = 'NewCity2'
    var_6 = [var_4, var_5]
    var_7 = {var_2: var_6}
    var_8 = [var_2]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'invalid_data'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'dict'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 12/17 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{st_num} {st_name}'
    var_5 = []
    var_6 = '123'
    var_7 = 'Main'
    var_8 = 'St'
    var_9 = 'en_US'
    var_10 = 'en_GB'
    var_11 = [var_9, var_10]
    var_12 = var_1.address()
    assert var_12 == '123 Main'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_correctly_for_shortened_locales. Retrieved 3/6 statements.
# Partially parsed test_address_formats_correctly_for_ja_locale. Retrieved 3/6 statements.
# Partially parsed test_address_formats_correctly_for_standard_locale. Retrieved 3/4 statements.
# Partially parsed test_address_includes_street_suffix_for_non_shortened_non_ja. Retrieved 4/5 statements.
# Partially parsed test_address_for_locale_with_shortened_format. Retrieved 4/7 statements.
# Partially parsed test_address_ja_locale_contains_city_and_numbers. Retrieved 5/8 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'st_num'
    var_5 = bool('st_num' not in var_3)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'st_sfx'
    var_5 = bool('st_sfx' not in var_3)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_number()
    var_3 = var_1.street_name()
    var_4 = var_1.address()
    var_5 = bool(var_2 in var_4)
    assert var_5 is True
    var_6 = bool(var_3 in var_4)
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_GB'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = var_2.street_suffix()
    var_5 = bool(var_4 in var_3)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = [var_0]
    var_2 = {}
    var_3 = module_0.Address(var_0, **var_2)
    var_4 = var_3.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'city'
    var_5 = [var_4]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'ja'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = var_2.address()
    var_7 = var_5.address()
    var_8 = bool(var_6 != var_7)
    assert var_8 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_locale_setup_before_dataset_load.




# Parsed testcases at query #11
#--------------------------

# Partially parsed test_address_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_shortened_locale. Retrieved 4/5 statements.
# Partially parsed test_address_ja_locale. Retrieved 4/5 statements.
# Partially parsed test_address_street_number_in_range. Retrieved 3/4 statements.
# Partially parsed test_address_street_name_present. Retrieved 3/4 statements.
# Partially parsed test_address_street_suffix_present. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_GB'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_locale_in_shortened_address_fmt. Retrieved 8/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = 'en_GB'
    var_2 = [var_0, var_1]
    var_3 = {}
    var_4 = module_0.Address(var_0, **var_3)
    var_5 = '{st_num} {st_name}'
    var_6 = '123'
    var_7 = 'Main'
    var_8 = var_4.address()
    assert var_8 == '123 Main'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 8/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '{st_num} {st_name}'
    var_3 = '123'
    var_4 = 'Main'
    var_5 = 'en_US'
    var_6 = 'en_GB'
    var_7 = [var_5, var_6]
    var_8 = var_1.address()
    assert var_8 == '123 Main'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_object.
# Partially parsed test_address_initialization_with_additional_args. Retrieved 4/5 statements.
# Partially parsed test_address_initialization_without_seed. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_locale_separator. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_number()
    var_6 = var_4.street_number()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'en'
    var_1 = 12345
    var_2 = 'extra_arg'
    var_3 = 'value'
    var_4 = [var_2]

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-US'
    var_4 = var_2._dataset

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.Meta.name
    assert var_2 == 'address'
    var_3 = var_1.Meta.datafile
    assert var_3 == 'address.json'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset != {})
    assert var_3 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_correctly_for_shortened_locale. Retrieved 3/5 statements.
# Partially parsed test_address_formats_correctly_for_ja_locale. Retrieved 3/4 statements.
# Partially parsed test_address_formats_correctly_for_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_contains_street_number_and_name. Retrieved 2/4 statements.
# Partially parsed test_address_uses_street_suffix_for_non_ja_non_shortened. Retrieved 3/4 statements.
# Partially parsed test_address_ja_locale_returns_string. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()



# Parsed testcases at query #17
#--------------------------






