####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_constructor. Retrieved 6/18 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1.locale
    var_4 = var_1._dataset
    var_5 = 12345
    var_6 = {}
    var_7 = module_0.Address(seed=var_5, **var_6)
    var_8 = var_7.locale
    var_9 = var_7._dataset
    var_10 = 42



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address. Retrieved 10/48 statements.


def test_case_0():
    var_0 = 'address'
    var_1 = 'street_number'
    var_2 = 'street_name'
    var_3 = 'street_suffix'
    var_4 = '_extract'
    var_5 = 'locale'
    var_6 = 'random'
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = 'Address'
    var_9 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_10 = '123'
    var_11 = 'Main'
    var_12 = 'St'



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_address_ja_locale. Retrieved 13/22 statements.


def test_case_0():
    var_0 = '123 Main St'
    var_1 = '456 Oak Ave'
    var_2 = [var_0, var_1]
    var_3 = 'Tokyo'
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = '{0}-{1}-{2}-{3}'
    var_9 = 'address_fmt'
    var_10 = [var_9]
    var_11 = 'Osaka'
    var_12 = [var_3, var_11]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address. Retrieved 4/17 statements.
# Partially parsed test_address_with_shortened_format. Retrieved 3/14 statements.
# Partially parsed test_address_japanese_locale. Retrieved 5/16 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Main'
    var_2 = 'Street'
    var_3 = '{st_num} {st_name} {st_sfx}'

def test_case_0():
    var_0 = '456'
    var_1 = 'Oxford'
    var_2 = '{st_num} {st_name}'

def test_case_0():
    var_0 = 'Tokyo'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_address_ja_locale. Retrieved 14/27 statements.


def test_case_0():
    var_0 = 'Tokyo'
    var_1 = 'Osaka'
    var_2 = 'Kyoto'
    var_3 = [var_0, var_1, var_2]
    var_4 = 1
    var_5 = 2
    var_6 = 3
    var_7 = [var_4, var_5, var_6]
    var_8 = '{} {} {} {}'
    var_9 = 'Tokyo'
    var_10 = 1
    var_11 = 2
    var_12 = 3
    var_13 = 'ja'



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address_with_shortened_format_locale. Retrieved 11/26 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'locale'
    var_1 = '_extract'
    var_2 = 'street_number'
    var_3 = 'street_name'
    var_4 = 'random'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = {}
    var_7 = module_0.Address(**var_6)
    var_8 = '{st_num} {st_name}'
    var_9 = '123'
    var_10 = 'Main St'
    var_11 = var_7.address()
    assert var_11 == '123 Main St'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address. Retrieved 21/29 statements.


def test_case_0():
    var_0 = 'address'
    var_1 = 'locale'
    var_2 = 'random'
    var_3 = '_extract'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = '123'
    var_6 = 'Main Street'
    var_7 = 'Ave'
    var_8 = '{st_num} {st_name} {st_sfx}'
    var_9 = 'address_fmt'
    var_10 = [var_9]
    var_11 = [var_7]
    var_12 = lambda keys: var_8 if keys == var_10 else var_11
    var_13 = 0
    var_14 = ''
    var_15 = lambda x: x[var_13] if x else var_14
    var_16 = 'address_fmt'
    var_17 = [var_16]
    var_18 = '123'
    var_19 = 'Main'
    var_20 = 'St'
    var_21 = '123'
    var_22 = 'Main'
    var_23 = 'St'



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_with_shortened_format_locale. Retrieved 4/16 statements.


def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '123'
    var_2 = 'Main'
    var_3 = 'St'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address. Retrieved 4/8 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'Test address method generates a valid address string.'
    var_1 = {}
    var_2 = module_0.Address(**var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_constructor. Retrieved 6/7 statements.
# Failed to parse test_address_constructor_with_locale.
# Partially parsed test_address_constructor_meta_attributes. Retrieved 3/7 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = 'locale'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = '_dataset'
    var_7 = hasattr(var_1, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = var_1._dataset
    var_10 = var_1.locale
    assert var_10 == 'en'

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
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = bool('street' in var_1._dataset or 'address_fmt' in var_1._dataset or 'city' in var_1._dataset)
    assert var_5 is True

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'name'
    var_3 = 'datafile'



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 10/29 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'locale'
    var_1 = '_extract'
    var_2 = 'street_number'
    var_3 = 'street_name'
    var_4 = 'random'
    var_5 = [var_0, var_1, var_2, var_3, var_4]
    var_6 = None
    var_7 = {}
    var_8 = module_0.Address(var_6, **var_7)
    var_9 = 0
    var_10 = var_8.address()
    var_11 = bool('456' in var_10 or var_10 == '456 Oak Ave')
    assert var_11 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_locale_ja_predicate. Retrieved 9/16 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'Test that the predicate at line 17 evaluates to True for Japanese locale.'
    var_1 = {}
    var_2 = module_0.Address(**var_1)
    var_3 = 'Tokyo'
    var_4 = [var_3]
    var_5 = 1
    var_6 = 2
    var_7 = 3
    var_8 = [var_5, var_6, var_7]
    var_9 = var_2.address()
    var_10 = var_2.locale
    assert var_10 == 'ja'



# Parsed testcases at query #15
#--------------------------






# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address. Retrieved 10/22 statements.
# Partially parsed test_address_shortened_format. Retrieved 9/19 statements.
# Partially parsed test_address_japanese_format. Retrieved 22/31 statements.


def test_case_0():
    var_0 = 'address'
    var_1 = 'street_number'
    var_2 = 'street_name'
    var_3 = 'street_suffix'
    var_4 = '_extract'
    var_5 = 'locale'
    var_6 = 'random'
    var_7 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6]
    var_8 = 'address_fmt'
    var_9 = [var_8]

def test_case_0():
    var_0 = 'address'
    var_1 = 'street_number'
    var_2 = 'street_name'
    var_3 = '_extract'
    var_4 = 'locale'
    var_5 = 'SHORTENED_ADDRESS_FMT'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'address_fmt'
    var_8 = [var_7]

def test_case_0():
    var_0 = 'address'
    var_1 = '_extract'
    var_2 = 'locale'
    var_3 = 'random'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'address_fmt'
    var_6 = [var_5]
    var_7 = 'city'
    var_8 = [var_7]
    var_9 = '{0} {1} {2} {3}'
    var_10 = 'Tokyo'
    var_11 = 'Osaka'
    var_12 = 'Kyoto'
    var_13 = [var_10, var_11, var_12]
    var_14 = {var_6: var_9, var_8: var_13}
    var_15 = []
    var_16 = 1
    var_17 = 2
    var_18 = 3
    var_19 = '{0} {1} {2} {3}'
    var_20 = [var_10, var_11, var_12]
    var_21 = 100



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_address_locale_ja_predicate. Retrieved 14/26 statements.


def test_case_0():
    var_0 = '123 Main St'
    var_1 = [var_0]
    var_2 = '123'
    var_3 = 'Main'
    var_4 = 'St'
    var_5 = 'Tokyo'
    var_6 = 1
    var_7 = 2
    var_8 = 3
    var_9 = [var_6, var_7, var_8]
    var_10 = '{} {} {} {}'
    var_11 = 'address_fmt'
    var_12 = [var_11]
    var_13 = [var_5]



# Parsed testcases at query #18
#--------------------------






# Parsed testcases at query #19
#--------------------------






# Parsed testcases at query #20
#--------------------------

# Partially parsed test_address_shortened_format. Retrieved 11/24 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '123'
    var_2 = 'Main St'
    var_3 = 'en_US'
    var_4 = 'de_DE'
    var_5 = [var_3, var_4]
    var_6 = {}
    var_7 = module_0.Address(**var_6)
    var_8 = '{st_num} {st_name}'
    var_9 = '123'
    var_10 = 'Main St'
    var_11 = var_7.address()
    assert var_11 == '123 Main St'
    var_12 = var_7.locale
    var_13 = bool(var_7.locale in var_5)
    assert var_13 is True



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_address_locale_ja. Retrieved 24/37 statements.


def test_case_0():
    var_0 = 'locale'
    var_1 = '_extract'
    var_2 = 'street_number'
    var_3 = 'street_name'
    var_4 = 'random'
    var_5 = 'address'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = 'Tokyo'
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'address_fmt'
    var_13 = [var_12]
    var_14 = 'Format: {} {} {} {}'
    var_15 = 'Osaka'
    var_16 = [var_7, var_15]
    var_17 = lambda keys: var_14 if keys == var_13 else var_16
    var_18 = [var_12]
    var_19 = '123'
    var_20 = 'Main'
    var_21 = 'city'
    var_22 = [var_21]
    var_23 = 100



# Parsed testcases at query #22
#--------------------------






# Parsed testcases at query #23
#--------------------------

# Partially parsed test_address. Retrieved 53/68 statements.


def test_case_0():
    var_0 = 'address'
    var_1 = 'locale'
    var_2 = 'random'
    var_3 = '_extract'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = lambda x: x[var_5]
    var_7 = 1
    var_8 = 2
    var_9 = 3
    var_10 = [var_7, var_8, var_9]
    var_11 = 'address_fmt'
    var_12 = [var_11]
    var_13 = 'street'
    var_14 = 'name'
    var_15 = [var_13, var_14]
    var_16 = 'suffix'
    var_17 = [var_13, var_16]
    var_18 = 'city'
    var_19 = [var_18]
    var_20 = '{st_num} {st_name} {st_sfx}'
    var_21 = 'Main'
    var_22 = 'Oak'
    var_23 = [var_21, var_22]
    var_24 = 'St'
    var_25 = 'Ave'
    var_26 = [var_24, var_25]
    var_27 = 'Tokyo'
    var_28 = [var_27]
    var_29 = {var_12: var_20, var_15: var_23, var_17: var_26, var_19: var_28}
    var_30 = []
    var_31 = lambda x: var_29.get(x, var_30)
    var_32 = ''
    var_33 = lambda x: x[var_5] if x else var_32
    var_34 = [var_7, var_8, var_9]
    var_35 = [var_11]
    var_36 = tuple(var_35)
    var_37 = [var_13, var_14]
    var_38 = tuple(var_37)
    var_39 = [var_13, var_16]
    var_40 = tuple(var_39)
    var_41 = 'Main Street'
    var_42 = [var_41]
    var_43 = [var_24]
    var_44 = {var_36: var_20, var_38: var_42, var_40: var_43}
    var_45 = []
    var_46 = lambda x: var_44.get(tuple(x), var_45)
    var_47 = '123'
    var_48 = 'Main'
    var_49 = 'St'
    var_50 = '{st_num} {st_name} {st_sfx}'
    var_51 = '123 Main St'
    var_52 = len(var_51)
    var_53 = bool(var_52 > 0)
    assert var_53 is True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_address. Retrieved 34/49 statements.


def test_case_0():
    var_0 = 'address'
    var_1 = 'locale'
    var_2 = 'random'
    var_3 = '_extract'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = ''
    var_7 = lambda x: x[var_5] if x else var_6
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'address_fmt'
    var_13 = [var_12]
    var_14 = '{st_num} {st_name} {st_sfx}'
    var_15 = 'street'
    var_16 = 'name'
    var_17 = [var_15, var_16]
    var_18 = '123 Main St'
    var_19 = [var_18]
    var_20 = 'suffix'
    var_21 = [var_15, var_20]
    var_22 = 'Street'
    var_23 = [var_22]
    var_24 = 'city'
    var_25 = [var_24]
    var_26 = 'City'
    var_27 = [var_26]
    var_28 = []
    var_29 = lambda x: var_14 if x == var_13 else var_19 if x == var_17 else var_23 if x == var_21 else var_27 if x == var_25 else var_28
    var_30 = '42'
    var_31 = 'Main Street'
    var_32 = 'Ave'
    var_33 = '{st_num} {st_name} {st_sfx}'



####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_constructor. Retrieved 6/7 statements.
# Failed to parse test_address_constructor_with_locale.
# Partially parsed test_address_constructor_with_locale_and_seed. Retrieved 1/6 statements.
# Partially parsed test_address_constructor_default_meta. Retrieved 2/3 statements.
# Partially parsed test_address_constructor_dataset_loaded. Retrieved 4/5 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = 'locale'
    var_4 = hasattr(var_1, var_3)
    var_5 = bool(var_4)
    assert var_5 is True
    var_6 = '_dataset'
    var_7 = hasattr(var_1, var_6)
    var_8 = bool(var_7)
    assert var_8 is True
    var_9 = var_1._dataset
    var_10 = var_1.locale
    assert var_10 == 'en'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = bool(var_2 is not None)
    assert var_3 is True
    var_4 = 'random'
    var_5 = hasattr(var_2, var_4)
    var_6 = bool(var_5)
    assert var_6 is True

def test_case_0():
    var_0 = 123

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'Meta'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True
    var_5 = var_1._dataset



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address. Retrieved 13/27 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = '_extract'
    var_1 = 'street_number'
    var_2 = 'street_name'
    var_3 = 'street_suffix'
    var_4 = 'locale'
    var_5 = 'random'
    var_6 = [var_0, var_1, var_2, var_3, var_4, var_5]
    var_7 = {}
    var_8 = module_0.Address(**var_7)
    var_9 = '{st_num} {st_name} {st_sfx}'
    var_10 = '123'
    var_11 = 'Main'
    var_12 = 'Street'
    var_13 = var_8.address()
    var_14 = '123'
    var_15 = bool('123' in var_13)
    assert var_15 is True
    var_16 = 'Main'
    var_17 = bool('Main' in var_13)
    assert var_17 is True
    var_18 = 'Street'
    var_19 = bool('Street' in var_13)
    assert var_19 is True



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_address_with_shortened_format_locale. Retrieved 6/20 statements.


def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '123'
    var_2 = 'Main St'
    var_3 = 'Street'
    var_4 = 'address_fmt'
    var_5 = [var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address. Retrieved 35/57 statements.


def test_case_0():
    var_0 = 'address'
    var_1 = 'locale'
    var_2 = 'random'
    var_3 = '_extract'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = ''
    var_7 = lambda x: x[var_5] if x else var_6
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'address_fmt'
    var_13 = [var_12]
    var_14 = 'street'
    var_15 = 'name'
    var_16 = [var_14, var_15]
    var_17 = 'suffix'
    var_18 = [var_14, var_17]
    var_19 = 'city'
    var_20 = [var_19]
    var_21 = '{st_num} {st_name} {st_sfx}'
    var_22 = 'Main Street'
    var_23 = 'Oak Avenue'
    var_24 = [var_22, var_23]
    var_25 = 'St'
    var_26 = 'Ave'
    var_27 = 'Rd'
    var_28 = [var_25, var_26, var_27]
    var_29 = 'Tokyo'
    var_30 = 'Osaka'
    var_31 = [var_29, var_30]
    var_32 = {var_13: var_21, var_16: var_24, var_18: var_28, var_20: var_31}
    var_33 = []
    var_34 = lambda keys: var_32.get(keys, var_33)



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_address_constructor. Retrieved 5/14 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1.locale
    var_4 = var_1._dataset
    var_5 = 12345
    var_6 = {}
    var_7 = module_0.Address(seed=var_5, **var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = var_7.locale
    var_10 = 54321



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address. Retrieved 48/75 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'locale'
    var_2 = 'random'
    var_3 = '_extract'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = ''
    var_7 = lambda x: x[var_5] if x else var_6
    var_8 = 123
    var_9 = 1
    var_10 = 2
    var_11 = 3
    var_12 = [var_9, var_10, var_11]
    var_13 = 'address_fmt'
    var_14 = [var_13]
    var_15 = 'street'
    var_16 = 'name'
    var_17 = [var_15, var_16]
    var_18 = tuple(var_17)
    var_19 = 'suffix'
    var_20 = [var_15, var_19]
    var_21 = tuple(var_20)
    var_22 = 'city'
    var_23 = [var_22]
    var_24 = tuple(var_23)
    var_25 = '{st_num} {st_name} {st_sfx}'
    var_26 = 'Main'
    var_27 = 'Oak'
    var_28 = [var_26, var_27]
    var_29 = 'St'
    var_30 = 'Ave'
    var_31 = [var_29, var_30]
    var_32 = 'Tokyo'
    var_33 = [var_32]
    var_34 = {var_14: var_25, var_18: var_28, var_21: var_31, var_24: var_33}
    var_35 = 0
    var_36 = ''
    var_37 = lambda x: x[var_35] if x else var_36
    var_38 = {}
    var_39 = module_0.Address(**var_38)
    var_40 = 'address_fmt'
    var_41 = [var_40]
    var_42 = '{st_num} {st_name} {st_sfx}'
    var_43 = lambda x: var_42 if x == var_41 else var_36
    var_44 = '123'
    var_45 = 'Main'
    var_46 = 'St'
    var_47 = var_39.address()
    var_48 = bool(var_10)
    assert var_48 is True
    var_49 = len(var_47)
    var_50 = bool(var_49 > 0)
    assert var_50 is True



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_with_shortened_format_locale. Retrieved 6/19 statements.


def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '123'
    var_2 = 'Main Street'
    var_3 = 'St'
    var_4 = 'address_fmt'
    var_5 = [var_4]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address. Retrieved 3/14 statements.
# Partially parsed test_address_shortened_format. Retrieved 3/12 statements.
# Partially parsed test_address_japan_locale. Retrieved 12/22 statements.


def test_case_0():
    var_0 = '123'
    var_1 = 'Main'
    var_2 = 'Street'
    var_3 = '123'
    var_4 = 'Main'
    var_5 = 'Street'

def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '42'
    var_2 = 'Hauptstraße'
    var_3 = '42'
    var_4 = 'Hauptstraße'

def test_case_0():
    var_0 = 'city'
    var_1 = [var_0]
    var_2 = 'Tokyo'
    var_3 = 'Osaka'
    var_4 = 'Kyoto'
    var_5 = [var_2, var_3, var_4]
    var_6 = '{0}-{1}-{2}-{3}'
    var_7 = lambda key: var_5 if key == var_1 else var_6
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address_locale_ja. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Tokyo'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '{0}-{1}-{2}-{3}'
    var_6 = 'address_fmt'
    var_7 = [var_6]
    var_8 = '123'
    var_9 = 'Main Street'
    var_10 = 'city'
    var_11 = [var_10]
    var_12 = 100



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_constructor. Retrieved 7/12 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1.locale
    var_4 = var_1._dataset
    var_5 = 12345
    var_6 = {}
    var_7 = module_0.Address(seed=var_5, **var_6)
    var_8 = bool(var_7 is not None)
    assert var_8 is True
    var_9 = var_7.locale
    var_10 = {}
    var_11 = module_0.Address(**var_10)
    var_12 = var_11._dataset
    var_13 = len(var_12)
    var_14 = bool(var_13 > 0)
    assert var_14 is True



# Parsed testcases at query #11
#--------------------------






# Parsed testcases at query #12
#--------------------------






# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_ja_locale. Retrieved 13/25 statements.


def test_case_0():
    var_0 = 'Tokyo'
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = [var_1, var_2, var_3]
    var_5 = '{0}-{1}-{2}-{3}'
    var_6 = 'address_fmt'
    var_7 = [var_6]
    var_8 = '123'
    var_9 = 'Main Street'
    var_10 = 'city'
    var_11 = [var_10]
    var_12 = 100



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_with_shortened_format_locale. Retrieved 7/18 statements.


def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '123'
    var_2 = 'Main St'
    var_3 = 'Ave'
    var_4 = '{st_num} {st_name}'
    var_5 = '123'
    var_6 = 'Main St'
    var_7 = bool(True)
    assert var_7 is True



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_address. Retrieved 25/39 statements.
# Partially parsed test_address_shortened_format. Retrieved 4/14 statements.
# Partially parsed test_address_japanese_locale. Retrieved 8/11 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'address'
    var_1 = 'locale'
    var_2 = 'random'
    var_3 = '_extract'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 0
    var_6 = ''
    var_7 = lambda x: x[var_5] if x else var_6
    var_8 = 1
    var_9 = 2
    var_10 = 3
    var_11 = [var_8, var_9, var_10]
    var_12 = 'address_fmt'
    var_13 = [var_12]
    var_14 = (var_12,)
    var_15 = '{st_num} {st_name} {st_sfx}'
    var_16 = {var_13: var_15, var_14: var_15}
    var_17 = 'Main'
    var_18 = lambda keys: var_16.get(keys, var_17)
    var_19 = 'en_US'
    var_20 = {}
    var_21 = module_0.Address(var_19, **var_20)
    var_22 = '123'
    var_23 = 'St'
    var_24 = var_21.address()
    var_25 = len(var_24)
    var_26 = bool(var_25 > 0)
    assert var_26 is True

def test_case_0():
    var_0 = 0
    var_1 = 'en_US'
    var_2 = '456'
    var_3 = 'Oak'

import mimesis.providers.address as module_0

def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 10
    var_4 = 20
    var_5 = 30
    var_6 = [var_3, var_4, var_5]
    var_7 = var_2.address()
    var_8 = len(var_7)
    var_9 = bool(var_8 > 0)
    assert var_9 is True



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address_init_dataset_empty_before_load. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'Test that _dataset is empty dict before _load_dataset is called.'



# Parsed testcases at query #17
#--------------------------






# Parsed testcases at query #18
#--------------------------

# Partially parsed test_address_with_shortened_format. Retrieved 7/26 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '{st_num} {st_name}'
    var_3 = '123'
    var_4 = 'Main St'
    var_5 = 'Street'
    var_6 = 'SHORTENED_ADDRESS_FMT'
    var_7 = var_1.address()
    var_8 = bool(var_6)
    assert var_8 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_address_with_shortened_format_locale. Retrieved 10/23 statements.


def test_case_0():
    var_0 = '{st_num} {st_name}'
    var_1 = '123'
    var_2 = 'Main St'
    var_3 = 'Street'
    var_4 = 'address_fmt'
    var_5 = [var_4]
    var_6 = 'en_US'
    var_7 = 'en_GB'
    var_8 = [var_6, var_7]
    var_9 = var_2 in var_8
    assert var_9 is True



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_address_constructor. Retrieved 7/18 statements.


import mimesis.providers.address as module_0

def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = bool(var_1 is not None)
    assert var_2 is True
    var_3 = var_1.locale
    var_4 = 12345
    var_5 = {}
    var_6 = module_0.Address(seed=var_4, **var_5)
    var_7 = bool(var_6 is not None)
    assert var_7 is True
    var_8 = var_6.locale
    var_9 = 54321
    var_10 = var_1._dataset
    var_11 = 'name'
    var_12 = 'datafile'



# Parsed testcases at query #21
#--------------------------






# Parsed testcases at query #22
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt_locale. Retrieved 6/19 statements.


def test_case_0():
    var_0 = '123 {st_num} {st_name}'
    var_1 = '456'
    var_2 = 'Main St'
    var_3 = 'Ave'
    var_4 = 'address_fmt'
    var_5 = [var_4]



