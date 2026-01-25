####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_with_gender_male.
# Failed to parse test_surname_with_gender_female.
# Partially parsed test_surname_with_gender_none. Retrieved 1/4 statements.
# Failed to parse test_surname_different_calls_return_different_values.
# Partially parsed test_surname_with_seed_returns_same_value. Retrieved 1/5 statements.
# Partially parsed test_surname_with_gender_and_seed. Retrieved 1/7 statements.
# Failed to parse test_surname_returns_valid_string_length.
# Failed to parse test_surname_with_different_genders_return_strings.
# Partially parsed test_surname_with_same_seed_and_gender_consistent. Retrieved 1/8 statements.
# Partially parsed test_surname_no_exception_with_valid_gender. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 123
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 999
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_email_default_domain. Retrieved 3/5 statements.
# Partially parsed test_email_custom_domains. Retrieved 4/7 statements.
# Partially parsed test_email_unique_without_seed. Retrieved 1/4 statements.
# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 2/5 statements.
# Partially parsed test_email_username_part. Retrieved 3/6 statements.
# Partially parsed test_email_domain_without_at_sign. Retrieved 4/9 statements.


def test_case_0():
    var_0 = []
    var_1 = '@'
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]

def test_case_0():
    var_0 = []
    var_1 = '@example.com'
    var_2 = '@test.org'
    var_3 = [var_1, var_2]
    var_4 = tuple(var_3)

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'You cannot use «unique» parameter with the seeded provider'

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '@'
    var_3 = email.split(var_2)[var_1]

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = 'test.org'
    var_3 = [var_1, var_2]
    var_4 = '@'



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_patronymic_with_gender_male.
# Failed to parse test_patronymic_with_gender_female.
# Failed to parse test_patronymic_without_gender.
# Partially parsed test_patronymic_with_none_gender. Retrieved 1/4 statements.
# Failed to parse test_patronymic_returns_string_or_none.
# Partially parsed test_patronymic_with_invalid_gender. Retrieved 1/4 statements.
# Partially parsed test_patronymic_seeded_randomness. Retrieved 1/7 statements.
# Partially parsed test_patronymic_different_genders_different_results. Retrieved 1/6 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'invalid'

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 123
    var_1 = []



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_with_gender_male.
# Failed to parse test_nationality_with_gender_female.
# Partially parsed test_nationality_with_gender_none. Retrieved 1/4 statements.
# Failed to parse test_nationality_with_gender_enum.
# Partially parsed test_nationality_different_calls_return_varied. Retrieved 7/10 statements.
# Partially parsed test_nationality_with_seed_returns_same. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 10
    var_2 = range(var_1)
    var_3 = [person.nationality() for _ in var_2]
    var_4 = 1
    var_5 = var_3[var_4:]
    var_6 = 0
    var_7 = var_3[var_6]

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []



# Parsed testcases at query #5
#--------------------------

# Partially parsed test_email_with_default_domain. Retrieved 3/5 statements.
# Partially parsed test_email_with_custom_domains. Retrieved 4/7 statements.
# Partially parsed test_email_unique_without_seed. Retrieved 1/4 statements.
# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 2/5 statements.
# Partially parsed test_email_without_unique_and_seed. Retrieved 1/5 statements.
# Partially parsed test_email_domain_without_at_symbol. Retrieved 9/12 statements.
# Partially parsed test_email_username_part. Retrieved 3/6 statements.


def test_case_0():
    var_0 = []
    var_1 = '@'
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]

def test_case_0():
    var_0 = []
    var_1 = '@test.com'
    var_2 = '@example.org'
    var_3 = [var_1, var_2]
    var_4 = tuple(var_3)

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = bool(True)
    assert var_4 is True

def test_case_0():
    var_0 = 67890
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 'test.com'
    var_2 = 'example.org'
    var_3 = [var_1, var_2]
    var_4 = 0
    var_5 = '@'
    var_6 = email.split(var_5)[var_4]
    var_7 = var_6 + var_5
    var_8 = 1
    var_9 = email.split(var_5)[var_8]
    var_10 = bool(var_9 in var_3)
    assert var_10 is True

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '@'
    var_3 = email.split(var_2)[var_1]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_custom_mask. Retrieved 5/15 statements.
# Partially parsed test_username_drange. Retrieved 4/7 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 4/7 statements.
# Partially parsed test_username_mask_without_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_separators. Retrieved 2/6 statements.
# Partially parsed test_username_lowercase_tag. Retrieved 1/4 statements.
# Partially parsed test_username_uppercase_tag. Retrieved 1/4 statements.
# Partially parsed test_username_capitalized_tag. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 1900
    var_2 = 1950
    var_3 = (var_1, var_2)
    var_4 = 'd'
    var_5 = var_3[0]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'd.d'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 'C-U.l'
    var_2 = '-.'

def test_case_0():
    var_0 = []
    var_1 = 'l'

def test_case_0():
    var_0 = []
    var_1 = 'U'

def test_case_0():
    var_0 = []
    var_1 = 'C'
    var_2 = 0
    var_3 = 1



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_email_default_domain. Retrieved 3/5 statements.
# Partially parsed test_email_custom_domains. Retrieved 5/9 statements.
# Partially parsed test_email_unique_without_seed. Retrieved 1/4 statements.
# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 2/5 statements.
# Partially parsed test_email_domain_without_at_symbol. Retrieved 3/6 statements.
# Partially parsed test_email_domain_with_at_symbol. Retrieved 2/5 statements.
# Partially parsed test_email_username_format. Retrieved 3/9 statements.
# Partially parsed test_email_no_domains_provided. Retrieved 4/6 statements.


def test_case_0():
    var_0 = []
    var_1 = '@'
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = 'test.org'
    var_3 = [var_1, var_2]
    var_4 = '@example.com'
    var_5 = '@test.org'

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'You cannot use «unique» parameter with the seeded provider'

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = [var_1]
    var_3 = '@example.com'

def test_case_0():
    var_0 = []
    var_1 = '@example.com'
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = 0
    var_2 = '@'
    var_3 = email.split(var_2)[var_1]

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_custom_mask. Retrieved 5/15 statements.
# Partially parsed test_username_with_different_separators. Retrieved 5/15 statements.
# Partially parsed test_username_drange_parameter. Retrieved 6/11 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 4/7 statements.
# Partially parsed test_username_mask_without_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_unique_outputs. Retrieved 3/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 1900
    var_2 = 1950
    var_3 = (var_1, var_2)
    var_4 = 'l_l_d'
    var_5 = '_'
    var_6 = 2
    var_7 = var_3[0]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = '###'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = 'l_d'
    var_3 = len(var_1)
    var_4 = bool(var_3 > 1)
    assert var_4 is True



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_custom_mask_c_c_d. Retrieved 5/15 statements.
# Partially parsed test_username_custom_mask_u_l_d. Retrieved 5/15 statements.
# Partially parsed test_username_custom_mask_l_l_d. Retrieved 8/20 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 4/7 statements.
# Partially parsed test_username_mask_without_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_mask_with_separators. Retrieved 1/5 statements.
# Partially parsed test_username_drange_custom. Retrieved 4/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 'l_l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)
    var_5 = '_'
    var_6 = 0
    var_7 = 1
    var_8 = 2
    var_9 = 1900

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = '###'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l).'

def test_case_0():
    var_0 = []
    var_1 = 'C-U.l_d'

def test_case_0():
    var_0 = []
    var_1 = 'd'
    var_2 = 5
    var_3 = 10
    var_4 = (var_2, var_3)
    var_5 = 5



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_patronymic_returns_none_when_patronymics_list_is_empty. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_gender_male.
# Failed to parse test_surname_with_gender_female.
# Partially parsed test_surname_with_gender_none. Retrieved 1/4 statements.
# Partially parsed test_surname_returns_string_from_surnames_list. Retrieved 4/7 statements.
# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 9/17 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'Smith'
    var_2 = 'Johnson'
    var_3 = 'Williams'
    var_4 = [var_1, var_2, var_3]

def test_case_0():
    var_0 = []
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

# Partially parsed test_surname_with_gender_separated_dict. Retrieved 9/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #14
#--------------------------

# Failed to parse test_patronymic_with_male_gender.
# Failed to parse test_patronymic_with_female_gender.
# Failed to parse test_patronymic_without_gender.
# Partially parsed test_patronymic_with_none_gender. Retrieved 1/4 statements.
# Failed to parse test_patronymic_returns_string_when_available.
# Failed to parse test_patronymic_returns_none_when_not_available.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #15
#--------------------------

# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_with_gender_male.
# Failed to parse test_nationality_with_gender_female.
# Partially parsed test_nationality_with_gender_none. Retrieved 1/4 statements.
# Partially parsed test_nationality_different_calls_return_different_values. Retrieved 5/6 statements.
# Partially parsed test_nationality_with_seed_returns_same_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 10
    var_2 = range(var_1)
    var_3 = [person.nationality() for _ in var_2]
    var_4 = set(var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 1)
    assert var_6 is True

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_with_gender_male.
# Failed to parse test_surname_with_gender_female.
# Partially parsed test_surname_with_gender_none. Retrieved 1/4 statements.
# Failed to parse test_surname_different_calls_return_different_values.
# Partially parsed test_surname_with_seed_returns_same_value. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 14/21 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'French'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = None
    var_9 = 'Random'
    var_10 = ()
    var_11 = 'choice'
    var_12 = 0
    var_13 = lambda lst: lst[var_12]
    var_14 = {var_11: var_13}
    var_15 = [var_9, var_10, var_14]



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nationality_with_gender_dict. Retrieved 13/20 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'American'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = 'Random'
    var_9 = ()
    var_10 = 'choice'
    var_11 = 0
    var_12 = lambda lst: lst[var_11]
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_nationality_with_dict_and_gender. Retrieved 13/20 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = [var_3]
    var_5 = 'French'
    var_6 = [var_5]
    var_7 = {var_1: var_4, var_2: var_6}
    var_8 = 'Random'
    var_9 = ()
    var_10 = 'choice'
    var_11 = 0
    var_12 = lambda lst: lst[var_11]
    var_13 = {var_10: var_12}
    var_14 = [var_8, var_9, var_13]



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_patronymic_returns_none_when_no_patronymics. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_surname_with_gender_separated_dict. Retrieved 9/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #22
#--------------------------

# Failed to parse test_patronymic_with_male_gender.
# Failed to parse test_patronymic_with_female_gender.
# Failed to parse test_patronymic_without_gender.
# Partially parsed test_patronymic_with_none_gender. Retrieved 1/4 statements.
# Partially parsed test_patronymic_returns_string_when_available. Retrieved 1/5 statements.
# Partially parsed test_patronymic_returns_none_when_not_available. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 'ru'
    var_1 = []

def test_case_0():
    var_0 = 'en'
    var_1 = []



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 9/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Russian'
    var_4 = 'American'
    var_5 = [var_3, var_4]
    var_6 = 'French'
    var_7 = 'Italian'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_surname_with_dict_surnames. Retrieved 9/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Partially parsed test_surname_with_none_gender. Retrieved 1/4 statements.
# Partially parsed test_surname_returns_string_from_surnames_list. Retrieved 2/9 statements.
# Partially parsed test_surname_uses_random_choice. Retrieved 1/6 statements.
# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 2/13 statements.
# Partially parsed test_surname_with_non_dict_surnames. Retrieved 2/8 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'surnames'
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = 42

def test_case_0():
    var_0 = []
    var_1 = 'surnames'
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = 'surnames'
    var_2 = [var_1]



####################################################################
#     TEST GENERATION BEGINS (DEEPMOSA + deepseek-chat t=0.8)      #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_username_default_mask. Retrieved 3/15 statements.
# Partially parsed test_username_custom_mask. Retrieved 5/18 statements.
# Partially parsed test_username_with_dots. Retrieved 5/18 statements.
# Partially parsed test_username_with_hyphen. Retrieved 5/18 statements.
# Partially parsed test_username_custom_drange. Retrieved 7/18 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 4/7 statements.
# Partially parsed test_username_mask_without_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_empty_mask. Retrieved 4/15 statements.
# Partially parsed test_username_complex_mask. Retrieved 1/4 statements.
# Partially parsed test_username_seeded_reproducible. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = '_'
    var_2 = 0
    var_3 = 1
    var_4 = 1800

def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 2
    var_6 = 1800

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'
    var_3 = 0
    var_4 = 1
    var_5 = 2
    var_6 = 1800

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'
    var_3 = 0
    var_4 = 1
    var_5 = 2
    var_6 = 1800

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)
    var_5 = '_'
    var_6 = 0
    var_7 = 1
    var_8 = 1900

def test_case_0():
    var_0 = []
    var_1 = 1800
    var_2 = 2100
    var_3 = 2200
    var_4 = (var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'd.d.d'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = ''
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 1800

def test_case_0():
    var_0 = []
    var_1 = 'C_l-U.d'
    var_2 = '.'
    var_3 = '-'
    var_4 = '_'

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'C_l_d'
    var_3 = []



# Parsed testcases at query #2
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_custom_mask. Retrieved 5/15 statements.
# Partially parsed test_username_with_dot_separator. Retrieved 5/15 statements.
# Partially parsed test_username_with_hyphen_separator. Retrieved 5/15 statements.
# Partially parsed test_username_custom_drange. Retrieved 6/11 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 4/7 statements.
# Partially parsed test_username_mask_without_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_empty_mask. Retrieved 1/5 statements.
# Partially parsed test_username_complex_mask. Retrieved 1/5 statements.
# Partially parsed test_username_seeded_reproducible. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 1900
    var_2 = 1950
    var_3 = (var_1, var_2)
    var_4 = 'l_d'
    var_5 = '_'
    var_6 = 1
    var_7 = var_3[0]

def test_case_0():
    var_0 = []
    var_1 = 1
    var_2 = 2
    var_3 = 3
    var_4 = (var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'd.d.d'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = ''

def test_case_0():
    var_0 = []
    var_1 = 'C_l-U.d'

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []
    var_3 = 'C_d'



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_patronymic_returns_string_or_none.
# Failed to parse test_patronymic_with_gender_male.
# Failed to parse test_patronymic_with_gender_female.
# Partially parsed test_patronymic_with_gender_none. Retrieved 1/4 statements.
# Failed to parse test_patronymic_returns_valid_patronymic.
# Failed to parse test_patronymic_for_locale_without_patronymics.
# Partially parsed test_patronymic_uses_random_choice. Retrieved 6/11 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 'Ivanovich'
    var_1 = 'Petrovich'
    var_2 = 'Sidorovich'
    var_3 = [var_0, var_1, var_2]
    var_4 = 0
    var_5 = var_3[var_4]



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_email_with_default_domain. Retrieved 3/5 statements.
# Partially parsed test_email_with_custom_domains. Retrieved 6/9 statements.
# Partially parsed test_email_unique_without_seed. Retrieved 1/4 statements.
# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 2/5 statements.
# Partially parsed test_email_contains_username_and_domain. Retrieved 1/5 statements.
# Partially parsed test_email_domain_starts_with_at. Retrieved 2/5 statements.
# Partially parsed test_email_domain_does_not_start_with_at. Retrieved 3/6 statements.
# Partially parsed test_email_unique_uses_uuid. Retrieved 6/9 statements.
# Partially parsed test_email_non_unique_uses_username. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = '@'
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = 'test.org'
    var_3 = [var_1, var_2]
    var_4 = '@example.com'
    var_5 = '@test.org'
    var_6 = (var_4, var_5)

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'You cannot use «unique» parameter with the seeded provider'

def test_case_0():
    var_0 = []
    var_1 = '@'

def test_case_0():
    var_0 = []
    var_1 = '@example.com'
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = [var_1]
    var_3 = '@example.com'

import uuid as module_0

def test_case_0():
    var_0 = []
    var_1 = True
    var_2 = 0
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = len(var_4)
    assert var_5 == 32
    var_6 = module_0.UUID(var_4)
    var_7 = bool(True)
    assert var_7 is True
    var_8 = bool(False)
    assert var_8 is True

def test_case_0():
    var_0 = []
    var_1 = False
    var_2 = '@'
    var_3 = email.split(var_2)[var_1]



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_person_constructor_without_arguments.
# Partially parsed test_person_constructor_with_locale. Retrieved 1/2 statements.
# Partially parsed test_person_constructor_with_seed. Retrieved 1/2 statements.
# Partially parsed test_person_constructor_with_locale_and_seed. Retrieved 2/3 statements.
# Partially parsed test_person_constructor_with_arbitrary_keyword_arguments. Retrieved 1/2 statements.
# Partially parsed test_person_constructor_with_arbitrary_positional_arguments. Retrieved 2/3 statements.
# Failed to parse test_person_constructor_meta_attributes.


def test_case_0():
    var_0 = 'en'
    var_1 = []

def test_case_0():
    var_0 = 12345
    var_1 = []

def test_case_0():
    var_0 = 'en'
    var_1 = 12345
    var_2 = []

def test_case_0():
    var_0 = 'value'
    var_1 = []

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]



# Parsed testcases at query #6
#--------------------------

# Partially parsed test_username_with_uppercase_tag. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'U'



# Parsed testcases at query #7
#--------------------------

# Failed to parse test_nationality_without_gender.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Failed to parse test_nationality_with_other_gender.
# Partially parsed test_nationality_with_none_gender. Retrieved 1/4 statements.
# Partially parsed test_nationality_returns_different_values. Retrieved 4/5 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 10
    var_2 = range(var_1)
    var_3 = {person.nationality() for _ in var_2}
    var_4 = len(var_3)
    var_5 = bool(var_4 > 1)
    assert var_5 is True



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_gender_male.
# Failed to parse test_surname_with_gender_female.
# Partially parsed test_surname_with_gender_none. Retrieved 1/4 statements.
# Partially parsed test_surname_returns_string_from_surnames_list. Retrieved 5/9 statements.
# Partially parsed test_surname_with_dict_surnames_and_gender_male. Retrieved 10/16 statements.
# Partially parsed test_surname_with_dict_surnames_and_gender_female. Retrieved 10/16 statements.
# Partially parsed test_surname_calls_random_choice. Retrieved 5/13 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'Smith'
    var_2 = 'Johnson'
    var_3 = 'Williams'
    var_4 = [var_1, var_2, var_3]
    var_5 = 0

def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 0

def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 1

def test_case_0():
    var_0 = []
    var_1 = 'Smith'
    var_2 = 'Johnson'
    var_3 = 'Williams'
    var_4 = [var_1, var_2, var_3]
    var_5 = False
    var_6 = bool(var_5)
    assert var_6 is True



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_person_constructor_without_arguments.
# Partially parsed test_person_constructor_with_locale. Retrieved 1/2 statements.
# Partially parsed test_person_constructor_with_seed. Retrieved 1/2 statements.
# Partially parsed test_person_constructor_with_locale_and_seed. Retrieved 2/3 statements.
# Partially parsed test_person_constructor_with_arbitrary_keyword_arguments. Retrieved 1/2 statements.
# Partially parsed test_person_constructor_with_arbitrary_positional_arguments. Retrieved 2/3 statements.
# Failed to parse test_person_constructor_meta_attribute.


def test_case_0():
    var_0 = 'en'
    var_1 = []

def test_case_0():
    var_0 = 12345
    var_1 = []

def test_case_0():
    var_0 = 'en'
    var_1 = 12345
    var_2 = []

def test_case_0():
    var_0 = 'value'
    var_1 = []

def test_case_0():
    var_0 = 'arg1'
    var_1 = 'arg2'
    var_2 = [var_0, var_1]



# Parsed testcases at query #10
#--------------------------

# Failed to parse test_patronymic_with_gender_male.
# Failed to parse test_patronymic_with_gender_female.
# Failed to parse test_patronymic_without_gender.
# Partially parsed test_patronymic_with_invalid_gender. Retrieved 1/4 statements.
# Failed to parse test_patronymic_returns_string_or_none.
# Failed to parse test_patronymic_with_locale_ru.
# Failed to parse test_patronymic_with_locale_uk.
# Failed to parse test_patronymic_with_non_supported_locale.


def test_case_0():
    var_0 = []
    var_1 = 'invalid'



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_surname_with_dict_surnames. Retrieved 14/19 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = 'Smith'
    var_3 = 'Johnson'
    var_4 = [var_2, var_3]
    var_5 = 'Williams'
    var_6 = 'Brown'
    var_7 = [var_5, var_6]
    var_8 = {var_0: var_4, var_1: var_7}
    var_9 = lambda keys: var_8
    var_10 = None
    var_11 = lambda gender, enum: var_0 if gender is var_10 else gender.value
    var_12 = 0
    var_13 = lambda seq: seq[var_12]
    var_14 = []



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_with_gender_male.
# Failed to parse test_nationality_with_gender_female.
# Partially parsed test_nationality_with_gender_none. Retrieved 1/4 statements.
# Failed to parse test_nationality_with_gender_enum.
# Failed to parse test_nationality_different_calls_return_different_values.
# Partially parsed test_nationality_with_seed_returns_same_value. Retrieved 1/5 statements.
# Partially parsed test_nationality_with_gender_and_seed. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 123
    var_1 = []
    var_2 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = False
    var_3 = True
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_username_default_mask. Retrieved 3/15 statements.
# Partially parsed test_username_custom_mask. Retrieved 5/17 statements.
# Partially parsed test_username_custom_drange. Retrieved 5/11 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 4/7 statements.
# Partially parsed test_username_mask_without_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_mask_with_separators. Retrieved 5/16 statements.
# Partially parsed test_username_mask_lowercase_only. Retrieved 2/9 statements.
# Partially parsed test_username_mask_uppercase_only. Retrieved 2/9 statements.
# Partially parsed test_username_mask_capitalized_only. Retrieved 3/10 statements.
# Partially parsed test_username_mask_mixed. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = '_'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 1800

def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'
    var_3 = 0
    var_4 = 1
    var_5 = 2

def test_case_0():
    var_0 = []
    var_1 = 1900
    var_2 = 2000
    var_3 = (var_1, var_2)
    var_4 = '_'
    var_5 = 1
    var_6 = 1900

def test_case_0():
    var_0 = []
    var_1 = 1800
    var_2 = 2100
    var_3 = 2200
    var_4 = (var_1, var_2, var_3)
    var_5 = bool(False)
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'd.d.d'
    var_2 = bool(False)
    assert var_2 is True

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'
    var_3 = '.'
    var_4 = 0
    var_5 = 1
    var_6 = 2

def test_case_0():
    var_0 = []
    var_1 = 'l_l_l'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U_U_U'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_C_C'
    var_2 = '_'
    var_3 = 0

def test_case_0():
    var_0 = []
    var_1 = 'C_l-U.d'
    var_2 = '-'
    var_3 = '.'
    var_4 = '_'



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_patronymic_returns_none_when_patronymics_is_empty. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = []



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_init_calls_super_init. Retrieved 10/18 statements.


def test_case_0():
    var_0 = 1
    var_1 = 2
    var_2 = 3
    var_3 = (var_0, var_1, var_2)
    var_4 = 'locale'
    var_5 = 'seed'
    var_6 = 'en_US'
    var_7 = 42
    var_8 = {var_4: var_6, var_5: var_7}
    var_9 = 'super'
    var_10 = [var_0, var_1, var_2]



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = False
    var_3 = True
    var_4 = True
    assert var_4 is True



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 4/6 statements.


def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = False
    var_3 = True
    var_4 = True
    var_5 = bool(var_4)
    assert var_5 is True



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_patronymic_returns_none_when_patronymics_list_is_empty. Retrieved 7/15 statements.


def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 'Random'
    var_3 = ()
    var_4 = 'choice'
    var_5 = 0
    var_6 = lambda lst: lst[var_5]
    var_7 = {var_4: var_6}
    var_8 = [var_2, var_3, var_7]



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_nationality_without_gender.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Partially parsed test_nationality_with_none_gender. Retrieved 1/5 statements.
# Partially parsed test_nationality_returns_different_values. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 1)
    assert var_3 is True



# Parsed testcases at query #21
#--------------------------

# Failed to parse test___init__.




# Parsed testcases at query #22
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 7/13 statements.


def test_case_0():
    var_0 = []
    var_1 = 'MALE'
    var_2 = 'FEMALE'
    var_3 = 'American'
    var_4 = 'British'
    var_5 = [var_3, var_4]
    var_6 = [var_3, var_4]
    var_7 = {var_1: var_5, var_2: var_6}



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_surname_with_gender_separated_dict. Retrieved 15/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 'Random'
    var_11 = ()
    var_12 = 'choice'
    var_13 = 0
    var_14 = lambda lst: lst[var_13]
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_surname_with_gender_separated_dict. Retrieved 15/23 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}
    var_10 = 'Random'
    var_11 = ()
    var_12 = 'choice'
    var_13 = 0
    var_14 = lambda lst: lst[var_13]
    var_15 = {var_12: var_14}
    var_16 = [var_10, var_11, var_15]



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_surname_with_gender_separated_dict. Retrieved 9/15 statements.


def test_case_0():
    var_0 = []
    var_1 = 'male'
    var_2 = 'female'
    var_3 = 'Smith'
    var_4 = 'Johnson'
    var_5 = [var_3, var_4]
    var_6 = 'Williams'
    var_7 = 'Brown'
    var_8 = [var_6, var_7]
    var_9 = {var_1: var_5, var_2: var_8}



