####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 3/4 statements.
# Partially parsed test_address_with_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_address_with_shortened_format. Retrieved 4/7 statements.


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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'en_US'
    var_3 = var_1.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_constructor_defaults. Retrieved 2/3 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset != {})
    assert var_5 is True



# Parsed testcases at query #3
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset != {})
    assert var_4 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address_ja_locale. Retrieved 2/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_address_street_suffix_included_for_non_shortened_locale. Retrieved 3/4 statements.
# Partially parsed test_address_street_suffix_not_included_for_shortened_locale. Retrieved 3/4 statements.
# Partially parsed test_address_returns_formatted_string. Retrieved 3/4 statements.
# Partially parsed test_address_uses_correct_format_for_ja_locale. Retrieved 7/14 statements.


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
    var_2 = var_1.address()
    var_3 = var_1.street_suffix()
    var_4 = bool(var_3 in var_2)
    assert var_4 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = var_1.street_suffix()
    var_4 = bool(var_3 not in var_2)
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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 'city'
    var_4 = [var_3]
    var_5 = 3
    var_6 = 1
    var_7 = 100



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1.seed



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_with_shortened_format. Retrieved 2/4 statements.
# Partially parsed test_address_with_ja_locale. Retrieved 2/6 statements.
# Partially parsed test_address_with_full_format. Retrieved 2/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = bool('st_num' in var_2 or 'st_name' in var_2)
    assert var_3 is True

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
    var_3 = bool('st_num' in var_2 or 'st_name' in var_2 or 'st_sfx' in var_2)
    assert var_3 is True



# Parsed testcases at query #8
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset != {})
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_locale_in_shortened_address_fmt. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 3/4 statements.
# Partially parsed test_address_with_custom_street_number. Retrieved 2/5 statements.
# Partially parsed test_address_with_custom_street_name. Retrieved 2/5 statements.
# Partially parsed test_address_with_custom_street_suffix. Retrieved 2/5 statements.
# Partially parsed test_address_with_shortened_format. Retrieved 3/5 statements.
# Partially parsed test_address_with_ja_locale. Retrieved 3/5 statements.


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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_locale_ja_predicate. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_locale_ja_predicate. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #13
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_locale_ja_predicate. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address_with_shortened_format. Retrieved 3/4 statements.
# Partially parsed test_address_with_japanese_locale. Retrieved 3/6 statements.
# Partially parsed test_address_with_default_locale. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = bool('st_num' in var_3 or 'st_name' in var_3)
    assert var_4 is True

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
    var_3 = var_2.address()
    var_4 = bool('st_num' in var_3 or 'st_name' in var_3 or 'st_sfx' in var_3)
    assert var_4 is True



# Parsed testcases at query #17
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #18
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_address. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_address_shortened_fmt_returns_correct_format. Retrieved 3/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'en_US'
    var_3 = var_1.address()
    var_4 = '{st_num}'
    var_5 = bool('{st_num}' in var_3)
    assert var_5 is True
    var_6 = '{st_name}'
    var_7 = bool('{st_name}' in var_3)
    assert var_7 is True
    var_8 = '{st_sfx}'
    var_9 = bool('{st_sfx}' not in var_3)
    assert var_9 is True



# Parsed testcases at query #21
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + mistralai/devstral-2512:free t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 3/7 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'en_US'
    var_3 = var_1.address()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_locale_in_shortened_address_fmt. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_locale_is_ja. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_address_constructor_defaults. Retrieved 2/3 statements.
# Partially parsed test_address_constructor_custom_locale. Retrieved 3/4 statements.
# Partially parsed test_address_constructor_custom_seed. Retrieved 3/4 statements.
# Partially parsed test_address_constructor_custom_locale_and_seed. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset != {})
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'es'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'es'
    var_5 = var_3._dataset
    var_6 = var_3._dataset
    var_7 = bool(var_3._dataset != {})
    assert var_7 is True



# Parsed testcases at query #6
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset != {})
    assert var_4 is True



# Parsed testcases at query #7
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(not var_1._dataset)
    assert var_2 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_default_initialization. Retrieved 2/3 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset != {})
    assert var_5 is True



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address_with_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_with_shortened_format_locale. Retrieved 3/5 statements.
# Partially parsed test_address_with_ja_locale. Retrieved 3/5 statements.


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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_with_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_with_shortened_format_locale. Retrieved 3/5 statements.
# Partially parsed test_address_with_ja_locale. Retrieved 3/5 statements.


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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 3/4 statements.
# Partially parsed test_address_with_custom_locale. Retrieved 3/5 statements.
# Partially parsed test_address_with_shortened_format. Retrieved 3/5 statements.


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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_locale_equals_ja. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 3/7 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'en_US'
    var_3 = var_1.address()



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_initialization_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_custom_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_seed. Retrieved 3/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset != {})
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_locale_equals_ja. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_locale_equals_ja. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'ja'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_address_constructor_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_constructor_custom_locale. Retrieved 3/4 statements.
# Partially parsed test_address_constructor_with_seed. Retrieved 3/4 statements.
# Partially parsed test_address_constructor_with_custom_locale_and_seed. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset
    var_4 = var_1._dataset
    var_5 = bool(var_1._dataset != {})
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'de'
    var_4 = var_2._dataset
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset
    var_5 = var_2._dataset
    var_6 = bool(var_2._dataset != {})
    assert var_6 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = 'es'
    var_2 = {}
    var_3 = module_0.Address(var_1, var_0, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'es'
    var_5 = var_3._dataset
    var_6 = var_3._dataset
    var_7 = bool(var_3._dataset != {})
    assert var_7 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_address_with_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_with_shortened_locale. Retrieved 3/5 statements.
# Partially parsed test_address_with_ja_locale. Retrieved 3/5 statements.


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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_address_with_shortened_format. Retrieved 3/6 statements.
# Partially parsed test_address_with_japanese_locale. Retrieved 2/6 statements.
# Partially parsed test_address_with_default_locale. Retrieved 2/4 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = ' '

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
    var_3 = ' '
    var_4 = bool(' ' in var_2)
    assert var_4 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_locale_in_shortened_address_fmt. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #21
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True



# Parsed testcases at query #22
#--------------------------




import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    var_3 = var_1._dataset
    var_4 = bool(var_1._dataset == {})
    assert var_4 is True



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_locale_in_shortened_address_fmt. Retrieved 1/2 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_locale_ja_returns_formatted_address. Retrieved 3/6 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = ' '



