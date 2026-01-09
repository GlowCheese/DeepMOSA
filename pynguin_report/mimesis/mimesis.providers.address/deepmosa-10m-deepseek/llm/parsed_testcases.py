####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_contains_street_number_and_name_for_shortened_locale. Retrieved 4/5 statements.
# Partially parsed test_address_uses_shortened_format_for_supported_locale. Retrieved 5/8 statements.
# Partially parsed test_address_uses_japanese_format_for_ja_locale. Retrieved 9/15 statements.
# Partially parsed test_address_uses_full_format_for_other_locales. Retrieved 5/8 statements.
# Partially parsed test_address_format_matches_extracted_template. Retrieved 14/23 statements.
# Partially parsed test_address_street_name_is_from_list. Retrieved 6/7 statements.
# Partially parsed test_address_street_suffix_is_from_list_for_non_shortened_locales. Retrieved 6/9 statements.
# Partially parsed test_address_for_ja_locale_contains_city. Retrieved 4/8 statements.
# Partially parsed test_address_for_ja_locale_contains_three_numbers. Retrieved 5/10 statements.


import mimesis.providers.address as module_0


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = var_1.street_suffix()
    var_6 = bool(var_5 not in var_2)
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = 'city'
    var_6 = [var_5]
    var_7 = 3
    var_8 = 1
    var_9 = 100


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = var_1.street_suffix()
    var_6 = bool(var_5 in var_2)
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = var_1.address()
    var_5 = var_1.street_number()
    var_6 = var_1.street_name()
    var_7 = 'city'
    var_8 = [var_7]
    var_9 = 3
    var_10 = 1
    var_11 = 100
    var_12 = var_1.street_number()
    var_13 = var_1.street_name()
    var_14 = var_1.street_suffix()
    var_15 = bool(var_4 == var_9)
    assert var_15 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = var_1.street_number()
    var_4 = int(var_3)
    var_5 = 1
    var_6 = bool(1 <= var_4)
    assert var_6 is True
    var_7 = bool(var_4 <= 1400)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'street'
    var_3 = 'name'
    var_4 = [var_2, var_3]
    var_5 = var_1.address()
    var_6 = var_1.street_name()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'street'
    var_3 = 'suffix'
    var_4 = [var_2, var_3]
    var_5 = var_1.address()
    var_6 = var_1.street_suffix()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'city'
    var_3 = [var_2]
    var_4 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 3
    var_4 = 1
    var_5 = 100



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_shortened_fmt_locale. Retrieved 4/5 statements.
# Partially parsed test_address_ja_locale. Retrieved 4/5 statements.
# Partially parsed test_address_street_number_in_range. Retrieved 3/4 statements.
# Partially parsed test_address_street_name_present. Retrieved 3/4 statements.
# Partially parsed test_address_street_suffix_present. Retrieved 4/5 statements.
# Partially parsed test_address_format_consistency. Retrieved 5/7 statements.
# Partially parsed test_address_no_empty_components. Retrieved 2/4 statements.
# Partially parsed test_address_contains_street_number. Retrieved 2/5 statements.
# Partially parsed test_address_contains_street_name. Retrieved 2/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True


def test_case_0():
    var_0 = 'en_US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = len(var_3)
    var_5 = bool(var_4 > 0)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = var_1.address()
    var_4 = len(var_2)
    var_5 = bool(var_4 > 0)
    assert var_5 is True
    var_6 = len(var_3)
    var_7 = bool(var_6 > 0)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = ''


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()



# Parsed testcases at query #3
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_object.
# Partially parsed test_address_initialization_without_datafile. Retrieved 2/5 statements.
# Partially parsed test_address_initialization_with_custom_datadir. Retrieved 11/24 statements.
# Partially parsed test_address_initialization_with_composite_locale. Retrieved 27/45 statements.
# Partially parsed test_address_initialization_with_args_and_kwargs. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_empty_dataset_for_locale. Retrieved 6/19 statements.
# Partially parsed test_address_initialization_with_non_dict_dataset_raises_error. Retrieved 7/21 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


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


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 'custom'
    var_1 = ''

def test_case_0():
    var_0 = 'en'
    var_1 = True
    var_2 = 'address.json'
    var_3 = 'street'
    var_4 = 'name'
    var_5 = 'Main St'
    var_6 = [var_5]
    var_7 = {var_4: var_6}
    var_8 = {var_3: var_7}
    var_9 = 'address'
    var_10 = 'address.json'

def test_case_0():
    var_0 = 'en'
    var_1 = 'en-US'
    var_2 = True
    var_3 = 'address.json'
    var_4 = 'street'
    var_5 = 'city'
    var_6 = 'name'
    var_7 = 'suffix'
    var_8 = 'Main St'
    var_9 = [var_8]
    var_10 = 'Street'
    var_11 = [var_10]
    var_12 = {var_6: var_9, var_7: var_11}
    var_13 = 'London'
    var_14 = [var_13]
    var_15 = {var_4: var_12, var_5: var_14}
    var_16 = 'Broadway'
    var_17 = [var_16]
    var_18 = {var_6: var_17}
    var_19 = {var_4: var_18}
    var_20 = 'address'
    var_21 = 'address.json'
    var_22 = [var_16]
    var_23 = [var_10]
    var_24 = {var_6: var_22, var_7: var_23}
    var_25 = [var_13]
    var_26 = {var_4: var_24, var_5: var_25}


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 is None

def test_case_0():
    var_0 = 'en'
    var_1 = 42
    var_2 = 'extra'
    var_3 = []


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'locale'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_dataset'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <de>'


def test_case_0():
    var_0 = 'zh-CN'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'zh-CN'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.Meta.name
    assert var_2 == 'address'
    var_3 = var_1.Meta.datafile
    assert var_3 == 'address.json'

def test_case_0():
    var_0 = 'xx'
    var_1 = True
    var_2 = 'address.json'
    var_3 = {}
    var_4 = 'address'
    var_5 = 'address.json'

def test_case_0():
    var_0 = 'en'
    var_1 = True
    var_2 = 'address.json'
    var_3 = 'not a dict'
    var_4 = 'address'
    var_5 = 'address.json'
    var_6 = 'en'
    var_7 = bool(False)
    assert var_7 is True
    var_8 = bool(True)
    assert var_8 is True



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_address_locale_in_shortened_address_fmt. Retrieved 8/11 statements.



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



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_locale_setup_before_dataset_load.




# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 13/19 statements.



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
    var_10 = [var_9]
    var_11 = var_1.address()
    assert var_11 == '123 Main'
    var_12 = [var_9]
    var_13 = var_1.address()
    assert var_13 == '123 Main St'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_with_street_number_and_name_for_shortened_locale. Retrieved 3/5 statements.
# Partially parsed test_address_ja_locale_returns_string. Retrieved 3/4 statements.
# Partially parsed test_address_default_locale_includes_street_suffix. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = 'st_num'
    var_5 = bool('st_num' not in var_3)
    assert var_5 is True


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_number()
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.street_name()
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.street_suffix()
    var_4 = var_2.address()
    var_5 = bool(var_3 in var_4)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = len(var_2)
    var_4 = bool(var_3 > 0)
    assert var_4 is True



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_locale_object. Retrieved 1/3 statements.
# Failed to parse test_address_initialization_with_missing_seed.
# Partially parsed test_address_initialization_with_additional_args. Retrieved 4/5 statements.
# Partially parsed test_address_initialization_with_locale_separator. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_locale_case_insensitivity. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_with_locale_constant.
# Partially parsed test_address_initialization_with_complex_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_ensure_dataset_is_dict. Retrieved 2/3 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


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

def test_case_0():
    var_0 = 'de'


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 42
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.seed
    assert var_3 == 42

def test_case_0():
    var_0 = 'en'
    var_1 = 123
    var_2 = 'extra_arg'
    var_3 = 'value'
    var_4 = [var_2]


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1.seed
    var_4 = bool(var_1.seed is not None)
    assert var_4 is True


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'ja'
    var_4 = var_2.seed
    var_5 = bool(var_2.seed is not None)
    assert var_5 is True


def test_case_0():
    var_0 = 999
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2.seed
    assert var_4 == 999


def test_case_0():
    var_0 = 'en-gb'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-gb'
    var_4 = var_2._dataset


def test_case_0():
    var_0 = 'EN'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'
    var_4 = var_2._dataset


def test_case_0():
    var_0 = 'es'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2._dataset
    var_4 = bool(var_2._dataset != {})
    assert var_4 is True
    var_5 = bool('street' in var_2._dataset or var_2._dataset == {})
    assert var_5 is True


def test_case_0():
    var_0 = ''
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = {}
    var_3 = module_0.Address(**var_2)
    var_4 = var_1.seed
    var_5 = bool(var_1.seed != var_3.seed)
    assert var_5 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.Meta.name
    assert var_2 == 'address'
    var_3 = var_1.Meta.datafile
    assert var_3 == 'address.json'


def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = 'test_seed'
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = hash(var_0)
    var_4 = var_2.seed
    var_5 = bool(var_2.seed == var_3)
    assert var_5 is True


def test_case_0():
    var_0 = 3.14
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = hash(var_0)
    var_4 = var_2.seed
    var_5 = bool(var_2.seed == var_3)
    assert var_5 is True


def test_case_0():
    var_0 = None
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = var_2.seed
    var_4 = bool(var_2.seed is not None)
    assert var_4 is True


def test_case_0():
    var_0 = 'zh-cn'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'zh-cn'
    var_4 = var_2._dataset


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_override_locale'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'override_locale'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <it>'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'get_current_locale'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'update_dataset'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_extract'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_load_dataset'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_setup_locale'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '_update_dict'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_locale_default_does_not_raise_unsupported_locale.




# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 7/10 statements.



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



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_get_current_locale. Retrieved 2/3 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


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


def test_case_0():
    var_0 = 'de'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'
    var_5 = var_3.street_number()
    var_6 = {}
    var_7 = module_0.Address(var_0, var_1, **var_6)
    var_8 = var_7.street_number()
    var_9 = bool(var_5 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'fr'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = var_2.default_country()
    var_7 = var_5.default_country()
    var_8 = bool(var_6 != var_7)
    assert var_8 is True


def test_case_0():
    var_0 = 'invalid'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset != {})
    assert var_3 is True
    var_4 = 'street'
    var_5 = bool('street' in var_1._dataset)
    assert var_5 is True


def test_case_0():
    var_0 = 'en'
    var_1 = 999
    var_2 = 'test'
    var_3 = 'extra_arg'
    var_4 = {var_3: var_2}
    var_5 = module_0.Address(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 == 'en'


def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <it>'


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_locale_setup_before_dataset_load. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 12/17 statements.



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



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 10/21 statements.


def test_case_0():
    var_0 = 'en'
    var_1 = 'en-AU'
    var_2 = 'en-CA'
    var_3 = 'en-GB'
    var_4 = 'en-IE'
    var_5 = 'en-IN'
    var_6 = 'en-NZ'
    var_7 = 'en-PH'
    var_8 = 'en-US'
    var_9 = [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_7, var_8]



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_locale_setup_before_dataset_load.




# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 4/8 statements.



def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 0
    var_4 = var_2.address()
    var_5 = bool(var_4 != '')
    assert var_5 is True



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_locale_setup_before_dataset_load. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'address'



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_correctly_for_shortened_locale. Retrieved 10/14 statements.
# Partially parsed test_address_formats_correctly_for_ja_locale. Retrieved 22/25 statements.
# Partially parsed test_address_formats_correctly_for_default_locale. Retrieved 12/17 statements.
# Partially parsed test_address_uses_street_number_and_name. Retrieved 9/12 statements.
# Partially parsed test_address_includes_street_suffix_when_not_shortened_or_ja. Retrieved 11/16 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


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
    var_9 = 'Random'
    var_10 = ()
    var_11 = 'choice'
    var_12 = 'randints'
    var_13 = 0
    var_14 = lambda x: x[var_13]
    var_15 = 1
    var_16 = 2
    var_17 = 3
    var_18 = [var_15, var_16, var_17]
    var_19 = lambda n, a, b: var_18
    var_20 = {var_11: var_14, var_12: var_19}
    var_21 = type(var_9, var_10, var_20)
    var_22 = var_2.address()
    assert var_22 == 'Tokyo123'


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = '{st_num} {st_name} {st_sfx}'
    var_6 = [var_5]
    var_7 = 'St'
    var_8 = 'Ave'
    var_9 = [var_7, var_8]
    var_10 = '456'
    var_11 = 'Oak'
    var_12 = var_2.address()
    assert var_12 == '456 Oak St'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'address_fmt'
    var_3 = [var_2]
    var_4 = '{st_num} {st_name}'
    var_5 = [var_4]
    var_6 = []
    var_7 = '789'
    var_8 = 'Pine'
    var_9 = var_1.address()
    assert var_9 == '789 Pine'


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'address_fmt'
    var_4 = [var_3]
    var_5 = '{st_num} {st_name} {st_sfx}'
    var_6 = [var_5]
    var_7 = 'Blvd'
    var_8 = [var_7]
    var_9 = '101'
    var_10 = 'Elm'
    var_11 = var_2.address()
    assert var_11 == '101 Elm Blvd'



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_with_locale_and_seed. Retrieved 4/5 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


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


def test_case_0():
    var_0 = 'de'
    var_1 = 42
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'
    var_5 = var_3._dataset


def test_case_0():
    var_0 = 'zh'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'zh'


def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'UnsupportedLocale'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset != {})
    assert var_3 is True


def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <it>'


def test_case_0():
    var_0 = 'en-gb'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-gb'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.seed
    var_3 = bool(var_1.seed is not None)
    assert var_3 is True



# Parsed testcases at query #3
#--------------------------





def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale


def test_case_0():
    var_0 = 'de'
    var_1 = 12345
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_locale_setup_before_dataset_load. Retrieved 1/9 statements.


def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_formats_correctly_for_shortened_locale. Retrieved 3/4 statements.
# Partially parsed test_address_formats_correctly_for_ja_locale. Retrieved 3/4 statements.
# Partially parsed test_address_formats_correctly_for_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_contains_street_number_and_name. Retrieved 2/4 statements.
# Partially parsed test_address_uses_street_suffix_for_non_ja_non_shortened. Retrieved 3/4 statements.
# Partially parsed test_address_for_locale_with_shortened_format. Retrieved 3/7 statements.
# Partially parsed test_address_for_ja_locale_specific_format. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()
    var_4 = bool('st_num' not in var_3 or 'st_name' not in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = 0
    var_4 = var_2.split()[var_3]
    var_5 = int(var_4)
    var_6 = 1
    var_7 = bool(1 <= var_5)
    assert var_7 is True
    var_8 = bool(var_5 <= 1400)
    assert var_8 is True


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

def test_case_0():
    var_0 = 'en_US'
    var_1 = 'en_GB'
    var_2 = [var_0, var_1]
    var_3 = bool(var_0)
    assert var_3 is True


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Partially parsed test_address_initialization_locale_affects_data. Retrieved 8/10 statements.
# Partially parsed test_address_initialization_locale_default. Retrieved 1/2 statements.
# Partially parsed test_address_initialization_locale_explicit. Retrieved 2/3 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


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


def test_case_0():
    var_0 = 'de'
    var_1 = 67890
    var_2 = {}
    var_3 = module_0.Address(var_0, var_1, **var_2)
    var_4 = var_3.locale
    assert var_4 == 'de'
    var_5 = var_3.street_number()
    var_6 = {}
    var_7 = module_0.Address(var_0, var_1, **var_6)
    var_8 = var_7.street_number()
    var_9 = bool(var_5 == var_8)
    assert var_9 is True


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'fr'
    var_4 = {}
    var_5 = module_0.Address(var_3, **var_4)
    var_6 = 'street'
    var_7 = 'name'
    var_8 = [var_6, var_7]
    var_9 = [var_6, var_7]


def test_case_0():
    var_0 = 'invalid_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1._dataset
    var_3 = bool(var_1._dataset != {})
    assert var_3 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.Meta.name
    assert var_2 == 'address'
    var_3 = var_1.Meta.datafile
    assert var_3 == 'address.json'


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = {}
    var_3 = module_0.Address(**var_2)
    var_4 = var_1.street_number()
    var_5 = var_3.street_number()
    var_6 = bool(var_4 != var_5)
    assert var_6 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)


def test_case_0():
    var_0 = 'ja'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)


def test_case_0():
    var_0 = 'es'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <es>'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 9/14 statements.



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



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_address_shortened_address_fmt. Retrieved 7/10 statements.



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



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 7/10 statements.



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



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_address_constructor_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_constructor_custom_locale. Retrieved 3/4 statements.
# Failed to parse test_address_constructor_locale_object.
# Partially parsed test_address_constructor_locale_with_region. Retrieved 3/4 statements.
# Partially parsed test_address_constructor_no_dataset_loading. Retrieved 2/5 statements.
# Partially parsed test_address_constructor_with_additional_args. Retrieved 5/6 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


def test_case_0():
    var_0 = 12345
    var_1 = {}
    var_2 = module_0.Address(seed=var_0, **var_1)
    var_3 = {}
    var_4 = module_0.Address(seed=var_0, **var_3)
    var_5 = var_2.street_name()
    var_6 = var_4.street_name()
    var_7 = bool(var_5 == var_6)
    assert var_7 is True


def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'UnsupportedLocale'


def test_case_0():
    var_0 = 'en-US'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-US'
    var_4 = var_2._dataset

def test_case_0():
    var_0 = 'address'
    var_1 = ''


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'random'
    var_3 = hasattr(var_1, var_2)
    var_4 = bool(var_3)
    assert var_4 is True
    var_5 = 'locale'
    var_6 = hasattr(var_1, var_5)
    var_7 = bool(var_6)
    assert var_7 is True
    var_8 = '_dataset'
    var_9 = hasattr(var_1, var_8)
    var_10 = bool(var_9)
    assert var_10 is True


def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <de>'


def test_case_0():
    var_0 = 'it'
    var_1 = 42
    var_2 = 'test'
    var_3 = 'extra_arg'
    var_4 = {var_3: var_2}
    var_5 = module_0.Address(var_0, var_1, **var_4)
    var_6 = var_5.locale
    assert var_6 == 'it'
    var_7 = var_5._dataset



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_contains_street_number_and_name. Retrieved 2/6 statements.
# Partially parsed test_address_locale_specific_shortened_format. Retrieved 2/6 statements.
# Partially parsed test_address_locale_specific_ja_format. Retrieved 2/6 statements.
# Partially parsed test_address_locale_specific_default_format. Retrieved 2/6 statements.
# Partially parsed test_address_uses_street_number_method. Retrieved 3/4 statements.
# Partially parsed test_address_uses_street_name_method. Retrieved 3/4 statements.
# Partially parsed test_address_uses_street_suffix_method_for_non_shortened_locales. Retrieved 3/7 statements.
# Partially parsed test_address_does_not_use_street_suffix_for_shortened_locales. Retrieved 3/7 statements.
# Partially parsed test_address_ja_locale_uses_city_and_randints. Retrieved 10/15 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = '123'
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'Main'
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'Strasse'
    var_3 = var_1.address()
    var_4 = bool(var_2 in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'ShouldNotAppear'
    var_3 = var_1.address()
    var_4 = bool(var_2 not in var_3)
    assert var_4 is True


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'Tokyo'
    var_3 = 'city'
    var_4 = [var_3]
    var_5 = ''
    var_6 = 10
    var_7 = 20
    var_8 = 30
    var_9 = [var_6, var_7, var_8]
    var_10 = var_1.address()
    var_11 = bool(var_2 in var_10)
    assert var_11 is True
    var_12 = '10'
    var_13 = bool('10' in var_10)
    assert var_13 is True
    var_14 = '20'
    var_15 = bool('20' in var_10)
    assert var_15 is True
    var_16 = '30'
    var_17 = bool('30' in var_10)
    assert var_17 is True



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_locale_is_not_setup_before_dataset_load. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'address'



# Parsed testcases at query #13
#--------------------------

# Failed to parse test_address_shortened_address_fmt.




# Parsed testcases at query #14
#--------------------------

# Partially parsed test_address_with_shortened_address_fmt. Retrieved 7/10 statements.



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



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_locale_not_supported_raises_unsupported_locale. Retrieved 3/8 statements.



def test_case_0():
    var_0 = 'unsupported_locale'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = 'Expected UnsupportedLocale to be raised'



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_address_returns_string. Retrieved 2/3 statements.
# Partially parsed test_address_contains_street_number_and_name. Retrieved 2/6 statements.
# Partially parsed test_address_locale_specific_shortened_format. Retrieved 4/7 statements.
# Partially parsed test_address_locale_ja_format. Retrieved 4/7 statements.
# Partially parsed test_address_uses_street_suffix_for_default_locale. Retrieved 3/4 statements.
# Partially parsed test_address_no_exception_for_valid_locale. Retrieved 3/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    var_4 = var_2.address()


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.address()
    var_3 = var_1.address()
    var_4 = bool(var_2 != var_3)
    assert var_4 is True


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.address()



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_address_initialization_with_default_locale. Retrieved 2/3 statements.
# Partially parsed test_address_initialization_with_specific_locale. Retrieved 3/4 statements.
# Failed to parse test_address_initialization_locale_object.
# Partially parsed test_address_initialization_with_locale_separator. Retrieved 3/4 statements.
# Partially parsed test_address_get_current_locale. Retrieved 2/3 statements.
# Partially parsed test_address_update_dataset. Retrieved 8/13 statements.
# Partially parsed test_address_update_dataset_with_invalid_data_raises_error. Retrieved 2/4 statements.



def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = var_1.locale
    assert var_2 == 'en'
    var_3 = var_1._dataset


def test_case_0():
    var_0 = 'fr'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'fr'
    var_4 = var_2._dataset


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


def test_case_0():
    var_0 = 'xx'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'UnsupportedLocale'


def test_case_0():
    var_0 = 'en-gb'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en-gb'
    var_4 = var_2._dataset


def test_case_0():
    var_0 = 'de'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = str(var_2)
    assert var_3 == 'Address <de>'


def test_case_0():
    var_0 = 'it'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)


def test_case_0():
    var_0 = 'en'
    var_1 = {}
    var_2 = module_0.Address(var_0, **var_1)
    var_3 = var_2.locale
    assert var_3 == 'en'


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


def test_case_0():
    var_0 = {}
    var_1 = module_0.Address(**var_0)
    var_2 = 'invalid_data'
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'dict'



