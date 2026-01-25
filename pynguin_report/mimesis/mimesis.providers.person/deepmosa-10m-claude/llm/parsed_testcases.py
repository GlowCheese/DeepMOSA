####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_is_not_empty.
# Partially parsed test_surname_multiple_calls_return_strings. Retrieved 4/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = range(var_1)
    var_3 = [person.surname() for _ in var_2]
    var_4 = 0



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 10/22 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = [var_6, var_8]
    var_10 = {var_3: var_7, var_4: var_9}



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_multiple_calls_return_strings.
# Partially parsed test_surname_with_seeded_provider. Retrieved 1/6 statements.
# Partially parsed test_surname_with_gender_and_seed. Retrieved 1/8 statements.


def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 123
    var_1 = []
    var_2 = []



# Parsed testcases at query #4
#--------------------------

# Partially parsed test_patronymic_with_male_gender. Retrieved 1/7 statements.
# Partially parsed test_patronymic_with_female_gender. Retrieved 1/7 statements.
# Partially parsed test_patronymic_with_none_gender. Retrieved 1/6 statements.
# Partially parsed test_patronymic_without_gender_parameter. Retrieved 1/6 statements.
# Failed to parse test_patronymic_returns_string_or_none.
# Partially parsed test_patronymic_with_seeded_provider. Retrieved 1/7 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = []



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_nationality_without_gender.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_not_empty.
# Failed to parse test_nationality_multiple_calls.




# Parsed testcases at query #6
#--------------------------

# Failed to parse test_username_with_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_with_hyphens. Retrieved 1/4 statements.
# Partially parsed test_username_with_underscore. Retrieved 1/4 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_dot_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_seeded_provider. Retrieved 2/6 statements.
# Partially parsed test_username_returns_string. Retrieved 1/5 statements.
# Partially parsed test_username_with_only_lowercase. Retrieved 1/5 statements.
# Partially parsed test_username_with_only_uppercase. Retrieved 1/5 statements.
# Partially parsed test_username_with_only_capitalized. Retrieved 2/7 statements.
# Partially parsed test_username_drange_with_large_values. Retrieved 4/7 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)
    var_5 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 'C_l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#.-_'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = []
    var_1 = 'l.C.d'
    var_2 = '.'

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = 'l_d'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = 'U_l_d'

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

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1800
    var_3 = 2100
    var_4 = (var_2, var_3)
    var_5 = '_'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 17/33 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = '_extract'
    var_3 = 'validate_enum'
    var_4 = 'random'
    var_5 = 'nationality'
    var_6 = [var_2, var_3, var_4, var_5]
    var_7 = 'male'
    var_8 = 'female'
    var_9 = 'Russian'
    var_10 = 'Ukrainian'
    var_11 = [var_9, var_10]
    var_12 = [var_9, var_10]
    var_13 = {var_7: var_11, var_8: var_12}
    var_14 = [var_5]
    var_15 = None
    var_16 = [var_5]



# Parsed testcases at query #8
#--------------------------

# Partially parsed test_username_with_uppercase_tag. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testuser'
    var_2 = 1950
    var_3 = 'U'
    var_4 = 'TESTUSER'



# Parsed testcases at query #9
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = '_extract'
    var_3 = 'validate_enum'
    var_4 = 'random'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'male'
    var_7 = 'female'
    var_8 = 'Russian'
    var_9 = 'Ukrainian'
    var_10 = [var_8, var_9]
    var_11 = [var_8, var_9]
    var_12 = 'nationality'
    var_13 = [var_12]



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = 'Brown'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_7, var_4: var_10}
    var_12 = 'surnames'
    var_13 = [var_12]
    var_14 = [var_5, var_6]



# Parsed testcases at query #11
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_with_lowercase_and_digits. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase. Retrieved 1/4 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_with_hyphen_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_underscore_separator. Retrieved 1/4 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_only_capitalized. Retrieved 1/4 statements.
# Partially parsed test_username_with_only_uppercase. Retrieved 1/4 statements.
# Partially parsed test_username_with_only_lowercase. Retrieved 1/4 statements.
# Partially parsed test_username_with_digit_only_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_multiple_separators. Retrieved 5/10 statements.
# Partially parsed test_username_with_custom_drange_large. Retrieved 4/7 statements.
# Partially parsed test_username_mask_none_generates_valid. Retrieved 1/5 statements.
# Partially parsed test_username_repeated_tags. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 'C_U_l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#-#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'must contain at least one of these'

def test_case_0():
    var_0 = []
    var_1 = 'C'

def test_case_0():
    var_0 = []
    var_1 = 'U'

def test_case_0():
    var_0 = []
    var_1 = 'l'

def test_case_0():
    var_0 = []
    var_1 = 'C_d'

def test_case_0():
    var_0 = []
    var_1 = 'C.U-l_d'
    var_2 = '.'
    var_3 = '-'
    var_4 = '_'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 2000
    var_3 = 2100
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = []
    var_1 = 'l_l_l_d'



# Parsed testcases at query #12
#--------------------------

# Failed to parse test_nationality_without_gender.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_not_empty.




# Parsed testcases at query #13
#--------------------------

# Partially parsed test_patronymic_with_male_gender. Retrieved 2/31 statements.
# Partially parsed test_patronymic_with_female_gender. Retrieved 2/31 statements.
# Partially parsed test_patronymic_with_none_gender. Retrieved 3/31 statements.
# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 2/25 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'

def test_case_0():
    var_0 = 'male'
    var_1 = 'female'

def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = None

def test_case_0():
    var_0 = 'male'
    var_1 = 'female'



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_patronymic_returns_none_when_patronymics_list_is_empty. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #16
#--------------------------

# Partially parsed test_username_uppercase_tag. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testname'
    var_2 = 1950
    var_3 = 'U'
    var_4 = 'TESTNAME'



# Parsed testcases at query #17
#--------------------------

# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 9/22 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = 'validate_enum'
    var_3 = '_extract'
    var_4 = 'random'
    var_5 = [var_2, var_3, var_4]
    var_6 = None
    var_7 = 'patronymic'
    var_8 = []



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nationality_with_dict_nationalities. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Russian'
    var_6 = 'German'
    var_7 = [var_5, var_6]
    var_8 = [var_5, var_6]
    var_9 = {var_3: var_7, var_4: var_8}



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 13/23 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = 'Smith'
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Johnson'
    var_6 = [var_2, var_5]
    var_7 = 'Williams'
    var_8 = 'Brown'
    var_9 = [var_7, var_8]
    var_10 = {var_3: var_6, var_4: var_9}
    var_11 = 'surnames'
    var_12 = [var_11]
    var_13 = 'male'
    var_14 = 'female'



# Parsed testcases at query #20
#--------------------------

# Partially parsed test_username_uppercase_tag_condition. Retrieved 3/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testname'
    var_2 = 1900
    var_3 = 'U'
    var_4 = 'TESTNAME'



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 9/23 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Russian'
    var_6 = 'German'
    var_7 = [var_5, var_6]
    var_8 = [var_5, var_6]
    var_9 = {var_3: var_7, var_4: var_8}



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 14/29 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = 'Smith'
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Johnson'
    var_6 = [var_2, var_5]
    var_7 = 'Williams'
    var_8 = [var_5, var_7]
    var_9 = {var_3: var_6, var_4: var_8}
    var_10 = 'surnames'
    var_11 = [var_10]
    var_12 = None
    var_13 = [var_10]



# Parsed testcases at query #23
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_multiple_separators. Retrieved 5/10 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_lowercase_tag. Retrieved 1/5 statements.
# Partially parsed test_username_with_uppercase_tag. Retrieved 1/5 statements.
# Partially parsed test_username_with_digit_tag. Retrieved 1/6 statements.
# Partially parsed test_username_complex_mask. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C.U-l_d'
    var_2 = '.'
    var_3 = '-'
    var_4 = '_'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 1950
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#-#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these'

def test_case_0():
    var_0 = []
    var_1 = 'l'

def test_case_0():
    var_0 = []
    var_1 = 'U'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'

def test_case_0():
    var_0 = []
    var_1 = 'C_U_l_d'
    var_2 = '_'



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_patronymic_with_male_gender. Retrieved 2/9 statements.
# Partially parsed test_patronymic_with_female_gender. Retrieved 2/9 statements.
# Partially parsed test_patronymic_with_none_gender. Retrieved 2/8 statements.
# Partially parsed test_patronymic_returns_string_or_none. Retrieved 1/5 statements.
# Partially parsed test_patronymic_with_unsupported_locale. Retrieved 1/4 statements.
# Partially parsed test_patronymic_multiple_calls_return_valid_types. Retrieved 1/7 statements.


def test_case_0():
    var_0 = 'ru'
    var_1 = []
    var_2 = None
    var_3 = [var_2]

def test_case_0():
    var_0 = 'ru'
    var_1 = []
    var_2 = None
    var_3 = [var_2]

def test_case_0():
    var_0 = 'ru'
    var_1 = []
    var_2 = None
    var_3 = [var_2]

def test_case_0():
    var_0 = 'ru'
    var_1 = []

def test_case_0():
    var_0 = 'en'
    var_1 = []

def test_case_0():
    var_0 = 'ru'
    var_1 = []



# Parsed testcases at query #25
#--------------------------

# Partially parsed test_username_predicate_line_48. Retrieved 7/13 statements.


def test_case_0():
    var_0 = 'Test that the predicate at line 48 (tag == "U") evaluates to True.'
    var_1 = 'testname'
    var_2 = 1900
    var_3 = 'U'
    var_4 = 'CUl'
    var_5 = 'U'
    var_6 = [var_5]
    var_7 = var_6[0]
    assert var_7 == 'U'
    var_8 = var_6[0]
    var_9 = bool(var_6[0] in var_4)
    assert var_9 is True



# Parsed testcases at query #26
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = 'Brown'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_7, var_4: var_10}



# Parsed testcases at query #27
#--------------------------

# Failed to parse test_patronymic_returns_none_when_empty.




# Parsed testcases at query #28
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Russian'
    var_6 = 'Ukrainian'
    var_7 = [var_5, var_6]
    var_8 = [var_5, var_6]
    var_9 = {var_3: var_7, var_4: var_8}



# Parsed testcases at query #29
#--------------------------

# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 2/5 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = []



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_username_uppercase_tag_condition. Retrieved 3/10 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testuser'
    var_2 = 1900
    var_3 = 'U'
    var_4 = 'TESTUSER'



# Parsed testcases at query #31
#--------------------------

# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Partially parsed test_surname_with_none_gender. Retrieved 1/6 statements.
# Failed to parse test_surname_multiple_calls_return_strings.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #32
#--------------------------

# Failed to parse test_nationality_without_gender.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_not_empty.




####################################################################
# TEST GENERATION BEGINS (DEEPMOSA + anthropic/claude-haiku-4.5 t=0.8) #
####################################################################


# Parsed testcases at query #1
#--------------------------

# Partially parsed test_patronymic_with_male_gender. Retrieved 8/17 statements.
# Partially parsed test_patronymic_with_female_gender. Retrieved 1/5 statements.
# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 6/14 statements.
# Failed to parse test_patronymic_returns_string.
# Failed to parse test_patronymic_with_gender_parameter.


def test_case_0():
    var_0 = 'patronymic'
    var_1 = 'validate_enum'
    var_2 = '_extract'
    var_3 = 'random'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = 'Ivanovich'
    var_6 = 'Petrovich'
    var_7 = []
    var_8 = None

def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 'patronymic'
    var_1 = 'validate_enum'
    var_2 = '_extract'
    var_3 = 'random'
    var_4 = [var_0, var_1, var_2, var_3]
    var_5 = []
    var_6 = None



# Parsed testcases at query #2
#--------------------------

# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #3
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_multiple_separators. Retrieved 5/10 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_invalid_drange_three_values. Retrieved 5/8 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_only_separators. Retrieved 1/4 statements.
# Partially parsed test_username_all_character_types. Retrieved 1/4 statements.
# Partially parsed test_username_repeated_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_digit_placeholder. Retrieved 2/5 statements.
# Partially parsed test_username_lowercase_only. Retrieved 1/8 statements.
# Partially parsed test_username_uppercase_only. Retrieved 1/4 statements.
# Partially parsed test_username_capitalized_only. Retrieved 1/4 statements.
# Partially parsed test_username_digit_only. Retrieved 1/5 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C.l-U_d'
    var_2 = '.'
    var_3 = '-'
    var_4 = '_'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 1950
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 1950
    var_4 = 2000
    var_5 = (var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#-#-#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = []
    var_1 = '.-_'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = []
    var_1 = 'C_U_l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'lll_ddd'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '#'

def test_case_0():
    var_0 = []
    var_1 = 'l'

def test_case_0():
    var_0 = []
    var_1 = 'U'

def test_case_0():
    var_0 = []
    var_1 = 'C'

def test_case_0():
    var_0 = []
    var_1 = 'd'



# Parsed testcases at query #4
#--------------------------

# Failed to parse test_username_with_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_with_hyphen_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_dot_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_underscore_separator. Retrieved 1/4 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_invalid_drange_three_values. Retrieved 5/8 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_only_digits. Retrieved 1/4 statements.
# Partially parsed test_username_with_seeded_provider. Retrieved 2/6 statements.
# Partially parsed test_username_mask_with_multiple_separators. Retrieved 1/4 statements.
# Partially parsed test_username_lowercase_only. Retrieved 1/5 statements.
# Partially parsed test_username_uppercase_only. Retrieved 1/5 statements.
# Partially parsed test_username_capitalized_only. Retrieved 2/7 statements.
# Partially parsed test_username_with_digits_only_part. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)
    var_5 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 'C.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'U_l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2000
    var_4 = 2021
    var_5 = (var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#.#.#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = []
    var_1 = 'd-d-d'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = 'l_d'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = 'C-U_l.d'

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

def test_case_0():
    var_0 = []
    var_1 = 'l_d_d'
    var_2 = '_'



# Parsed testcases at query #5
#--------------------------

# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Partially parsed test_nationality_with_none_gender. Retrieved 1/6 statements.
# Failed to parse test_nationality_multiple_calls_return_strings.
# Partially parsed test_nationality_with_seed. Retrieved 1/6 statements.
# Partially parsed test_nationality_with_seed_and_gender. Retrieved 1/8 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []



# Parsed testcases at query #6
#--------------------------

# Failed to parse test_email_default.
# Partially parsed test_email_with_custom_domains. Retrieved 3/9 statements.
# Partially parsed test_email_with_single_custom_domain. Retrieved 2/6 statements.
# Partially parsed test_email_unique. Retrieved 1/7 statements.
# Partially parsed test_email_unique_with_seeded_provider_raises_error. Retrieved 2/6 statements.
# Partially parsed test_email_format. Retrieved 3/12 statements.
# Partially parsed test_email_with_domain_without_at_symbol. Retrieved 2/5 statements.
# Partially parsed test_email_with_domain_with_at_symbol. Retrieved 2/5 statements.
# Partially parsed test_email_multiple_calls_different_results. Retrieved 6/12 statements.


def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = 'test.org'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = 'custom.com'
    var_2 = [var_1]
    var_3 = '@custom.com'

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'seeded'
    var_5 = bool('seeded' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = '@'
    var_2 = 0
    var_3 = 1

def test_case_0():
    var_0 = []
    var_1 = 'gmail.com'
    var_2 = [var_1]
    var_3 = '@gmail.com'

def test_case_0():
    var_0 = []
    var_1 = '@yahoo.com'
    var_2 = [var_1]
    var_3 = '@yahoo.com'

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = range(var_1)
    var_3 = [person.email() for _ in var_2]
    var_4 = set(var_3)
    var_5 = len(var_4)
    var_6 = bool(var_5 > 1)
    assert var_6 is True
    var_7 = '@'



# Parsed testcases at query #7
#--------------------------

# Partially parsed test_patronymic_with_male_gender. Retrieved 2/32 statements.
# Partially parsed test_patronymic_with_female_gender. Retrieved 2/31 statements.
# Partially parsed test_patronymic_with_none_gender. Retrieved 3/31 statements.
# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 2/31 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'

def test_case_0():
    var_0 = 'male'
    var_1 = 'female'

def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = None

def test_case_0():
    var_0 = 'male'
    var_1 = 'female'



# Parsed testcases at query #8
#--------------------------

# Failed to parse test_surname_with_no_gender.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Failed to parse test_surname_returns_string_type.
# Failed to parse test_surname_multiple_calls_return_strings.
# Partially parsed test_surname_with_seeded_provider. Retrieved 1/6 statements.


def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = []



# Parsed testcases at query #9
#--------------------------

# Failed to parse test_username_with_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_multiple_separators. Retrieved 5/10 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_with_invalid_drange. Retrieved 3/6 statements.
# Partially parsed test_username_with_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_mask_with_all_tag_types. Retrieved 1/4 statements.
# Partially parsed test_username_with_only_lowercase. Retrieved 1/5 statements.
# Partially parsed test_username_with_only_uppercase. Retrieved 1/5 statements.
# Partially parsed test_username_with_only_capitalized. Retrieved 2/7 statements.
# Partially parsed test_username_with_digits_only. Retrieved 1/4 statements.
# Partially parsed test_username_with_drange_reversed. Retrieved 4/7 statements.
# Partially parsed test_username_with_dot_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_hyphen_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_underscore_separator. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C.l-U_d'
    var_2 = '.'
    var_3 = '-'
    var_4 = '_'
    var_5 = [var_2, var_3, var_4]

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2000
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#-#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these'

def test_case_0():
    var_0 = []
    var_1 = 'C_U_l_d'
    var_2 = '_'

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

def test_case_0():
    var_0 = []
    var_1 = 'l_d'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 2100
    var_3 = 1800
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 'l_l_d'
    var_2 = '_'



# Parsed testcases at query #10
#--------------------------

# Partially parsed test_nationality_with_dict_nationalities. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = '_extract'
    var_3 = 'validate_enum'
    var_4 = 'random'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'male'
    var_7 = 'female'
    var_8 = 'Russian'
    var_9 = 'Ukrainian'
    var_10 = [var_8, var_9]
    var_11 = [var_8, var_9]
    var_12 = 'nationality'
    var_13 = [var_12]



# Parsed testcases at query #11
#--------------------------

# Partially parsed test_email_generates_valid_email. Retrieved 3/6 statements.
# Partially parsed test_email_with_custom_domains. Retrieved 3/8 statements.
# Partially parsed test_email_with_domain_without_at_symbol. Retrieved 2/4 statements.
# Partially parsed test_email_with_domain_with_at_symbol. Retrieved 2/4 statements.
# Partially parsed test_email_unique_generates_different_emails. Retrieved 1/4 statements.
# Partially parsed test_email_unique_with_seeded_provider_raises_error. Retrieved 2/5 statements.
# Partially parsed test_email_format_with_username_and_domain. Retrieved 4/10 statements.
# Failed to parse test_email_default_domains.


def test_case_0():
    var_0 = []
    var_1 = '@'
    var_2 = 1
    var_3 = '@'
    var_4 = email.split(var_3)[var_2]
    var_5 = '.'
    var_6 = bool('.' in var_4)
    assert var_6 is True

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = 'test.org'
    var_3 = [var_1, var_2]

def test_case_0():
    var_0 = []
    var_1 = 'example.com'
    var_2 = [var_1]
    var_3 = '@example.com'

def test_case_0():
    var_0 = []
    var_1 = '@example.com'
    var_2 = [var_1]
    var_3 = '@example.com'

def test_case_0():
    var_0 = []
    var_1 = True

def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True
    var_4 = 'unique'
    var_5 = bool('unique' in str(e).lower())
    assert var_5 is True

def test_case_0():
    var_0 = []
    var_1 = 'test.com'
    var_2 = [var_1]
    var_3 = '@'
    var_4 = 0



# Parsed testcases at query #12
#--------------------------

# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 4/13 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = 'male'
    var_3 = []



# Parsed testcases at query #13
#--------------------------

# Partially parsed test_email_raises_value_error_when_unique_true_and_seeded. Retrieved 2/5 statements.


def test_case_0():
    var_0 = 12345
    var_1 = []
    var_2 = True
    var_3 = bool(False)
    assert var_3 is True



# Parsed testcases at query #14
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 11/23 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = 'Brown'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_7, var_4: var_10}



# Parsed testcases at query #15
#--------------------------

# Partially parsed test_email_unique_with_seed_raises_value_error. Retrieved 7/12 statements.


def test_case_0():
    var_0 = '_has_seed'
    var_1 = 'email'
    var_2 = [var_0, var_1]
    var_3 = True
    var_4 = True
    var_5 = True
    var_6 = var_4 and var_5
    assert var_6 is True



# Parsed testcases at query #16
#--------------------------

# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Partially parsed test_surname_with_none_gender. Retrieved 1/5 statements.
# Partially parsed test_surname_returns_consistent_result_with_seed. Retrieved 1/5 statements.
# Partially parsed test_surname_returns_different_results_without_seed. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = set()
    var_2 = len(var_1)
    var_3 = bool(var_2 > 1)
    assert var_3 is True



# Parsed testcases at query #17
#--------------------------

# Failed to parse test_username_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_invalid_drange_three_elements. Retrieved 5/8 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_with_hyphen_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_dot_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_underscore_separator. Retrieved 1/4 statements.
# Partially parsed test_username_only_lowercase. Retrieved 1/4 statements.
# Partially parsed test_username_only_uppercase. Retrieved 1/4 statements.
# Partially parsed test_username_only_capitalized. Retrieved 1/4 statements.
# Partially parsed test_username_multiple_digits. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'C_C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2000
    var_4 = 2100
    var_5 = (var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#-#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = []
    var_1 = 'l-l-d'
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 'l.l.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'l_l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l'

def test_case_0():
    var_0 = []
    var_1 = 'U'

def test_case_0():
    var_0 = []
    var_1 = 'C'

def test_case_0():
    var_0 = []
    var_1 = 'l_dd'



# Parsed testcases at query #18
#--------------------------

# Partially parsed test_nationality. Retrieved 1/6 statements.
# Partially parsed test_nationality_with_gender. Retrieved 2/7 statements.
# Partially parsed test_nationality_returns_string. Retrieved 1/4 statements.
# Partially parsed test_nationality_not_empty. Retrieved 1/3 statements.


def test_case_0():
    var_0 = 'en'
    var_1 = []

def test_case_0():
    var_0 = 'en'
    var_1 = []
    var_2 = None

def test_case_0():
    var_0 = 'en'
    var_1 = []

def test_case_0():
    var_0 = 'en'
    var_1 = []



# Parsed testcases at query #19
#--------------------------

# Partially parsed test_nationality_with_gender_separated_dict. Retrieved 9/21 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Russian'
    var_6 = 'Ukrainian'
    var_7 = [var_5, var_6]
    var_8 = [var_5, var_6]
    var_9 = {var_3: var_7, var_4: var_8}



# Parsed testcases at query #20
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Failed to parse test_surname_returns_string.
# Failed to parse test_surname_not_empty.
# Partially parsed test_surname_with_seed. Retrieved 1/5 statements.
# Partially parsed test_surname_different_calls. Retrieved 4/9 statements.


def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = 5
    var_2 = range(var_1)
    var_3 = [person.surname() for _ in var_2]
    var_4 = 0



# Parsed testcases at query #21
#--------------------------

# Partially parsed test_username_uppercase_tag_predicate. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testname'
    var_2 = 1900
    var_3 = 'U'
    var_4 = 'TESTNAME'



# Parsed testcases at query #22
#--------------------------

# Partially parsed test_username_uppercase_tag_condition. Retrieved 12/18 statements.


import re as module_0

def test_case_0():
    var_0 = 'random'
    var_1 = [var_0]
    var_2 = 'testname'
    var_3 = 1950
    var_4 = 'U'
    var_5 = 1800
    var_6 = 2100
    var_7 = (var_5, var_6)
    var_8 = 'CUl'
    var_9 = '[CUld.\\-_]'
    var_10 = module_0.findall(var_9, var_4)
    var_11 = 'U'
    var_12 = bool('U' in var_10)
    assert var_12 is True
    var_13 = 'U'
    assert var_13 == 'U'



# Parsed testcases at query #23
#--------------------------

# Partially parsed test_email_raises_value_error_when_unique_and_seeded. Retrieved 4/8 statements.


def test_case_0():
    var_0 = '_has_seed'
    var_1 = 'email'
    var_2 = [var_0, var_1]
    var_3 = True



# Parsed testcases at query #24
#--------------------------

# Partially parsed test_nationality_with_dict_nationalities. Retrieved 14/23 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = '_extract'
    var_3 = 'validate_enum'
    var_4 = 'random'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'male'
    var_7 = 'female'
    var_8 = 'Russian'
    var_9 = 'American'
    var_10 = [var_8, var_9]
    var_11 = [var_8, var_9]
    var_12 = 'nationality'
    var_13 = [var_12]



# Parsed testcases at query #25
#--------------------------

# Failed to parse test_patronymic_returns_none_when_empty.




# Parsed testcases at query #26
#--------------------------

# Failed to parse test_surname_without_gender.
# Failed to parse test_surname_with_male_gender.
# Failed to parse test_surname_with_female_gender.
# Failed to parse test_surname_returns_string.
# Partially parsed test_surname_with_none_gender. Retrieved 1/5 statements.
# Partially parsed test_surname_consistency_with_seed. Retrieved 1/5 statements.
# Partially parsed test_surname_different_without_seed. Retrieved 11/13 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []

def test_case_0():
    var_0 = []
    var_1 = []
    var_2 = 10
    var_3 = range(var_2)
    var_4 = [person1.surname() for _ in var_3]
    var_5 = len(var_4)
    var_6 = set(var_4)
    var_7 = len(var_6)
    var_8 = var_5 == var_7
    var_9 = set(var_4)
    var_10 = len(var_9)
    var_11 = 1
    var_12 = var_10 > var_11
    var_13 = bool(var_8 or var_12)
    assert var_13 is True



# Parsed testcases at query #27
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 11/24 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = 'Brown'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_7, var_4: var_10}



# Parsed testcases at query #28
#--------------------------

# Partially parsed test_patronymic_with_male_gender. Retrieved 1/8 statements.
# Partially parsed test_patronymic_with_female_gender. Retrieved 1/8 statements.
# Partially parsed test_patronymic_without_gender. Retrieved 1/7 statements.
# Failed to parse test_patronymic_returns_string_or_none.
# Partially parsed test_patronymic_with_seed. Retrieved 1/8 statements.


def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = []
    var_1 = None
    var_2 = [var_1]

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []



# Parsed testcases at query #29
#--------------------------

# Failed to parse test_username_with_default_mask.
# Partially parsed test_username_with_lowercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_uppercase_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_capitalized_mask. Retrieved 1/4 statements.
# Partially parsed test_username_with_custom_drange. Retrieved 4/7 statements.
# Partially parsed test_username_with_dots_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_dash_separator. Retrieved 1/4 statements.
# Partially parsed test_username_with_underscore_separator. Retrieved 1/4 statements.
# Partially parsed test_username_invalid_drange_length. Retrieved 3/6 statements.
# Partially parsed test_username_invalid_drange_too_many. Retrieved 5/8 statements.
# Partially parsed test_username_no_required_tags. Retrieved 1/4 statements.
# Partially parsed test_username_complex_mask. Retrieved 1/4 statements.
# Partially parsed test_username_only_lowercase. Retrieved 1/5 statements.
# Partially parsed test_username_only_uppercase. Retrieved 1/5 statements.
# Partially parsed test_username_with_seeded_provider. Retrieved 2/6 statements.
# Partially parsed test_username_multiple_digits. Retrieved 2/6 statements.


def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'U_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'C_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2021
    var_4 = (var_2, var_3)

def test_case_0():
    var_0 = []
    var_1 = 'l.U.d'
    var_2 = '.'

def test_case_0():
    var_0 = []
    var_1 = 'l-U-d'
    var_2 = '-'

def test_case_0():
    var_0 = []
    var_1 = 'l_U_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = (var_2,)
    var_4 = bool(False)
    assert var_4 is True
    var_5 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = 'l_d'
    var_2 = 1900
    var_3 = 2000
    var_4 = 2100
    var_5 = (var_2, var_3, var_4)
    var_6 = bool(False)
    assert var_6 is True
    var_7 = 'drange parameter must contain only two integers'

def test_case_0():
    var_0 = []
    var_1 = '#-#'
    var_2 = bool(False)
    assert var_2 is True
    var_3 = 'Username mask must contain at least one of these: (C, U, l)'

def test_case_0():
    var_0 = []
    var_1 = 'C_l_U_d'
    var_2 = '_'

def test_case_0():
    var_0 = []
    var_1 = 'l'

def test_case_0():
    var_0 = []
    var_1 = 'U'

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = 'l_d'
    var_3 = []

def test_case_0():
    var_0 = []
    var_1 = 'l_d_d_d'
    var_2 = '_'



# Parsed testcases at query #30
#--------------------------

# Partially parsed test_nationality_with_dict_nationalities. Retrieved 13/22 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = '_extract'
    var_3 = 'validate_enum'
    var_4 = 'random'
    var_5 = [var_2, var_3, var_4]
    var_6 = 'Russian'
    var_7 = 'male'
    var_8 = 'female'
    var_9 = 'American'
    var_10 = [var_6, var_9]
    var_11 = [var_6, var_9]
    var_12 = {var_7: var_10, var_8: var_11}



# Parsed testcases at query #31
#--------------------------

# Partially parsed test_username_uppercase_tag. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testname'
    var_2 = 1950
    var_3 = 'U'
    var_4 = 'TESTNAME'



# Parsed testcases at query #32
#--------------------------

# Partially parsed test_nationality_with_dict_nationalities. Retrieved 12/25 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = 'male'
    var_3 = 'female'
    var_4 = 'Russian'
    var_5 = 'American'
    var_6 = [var_4, var_5]
    var_7 = [var_4, var_5]
    var_8 = {var_2: var_6, var_3: var_7}
    var_9 = 'nationality'
    var_10 = [var_9]
    var_11 = None



# Parsed testcases at query #33
#--------------------------

# Partially parsed test_patronymic_returns_none_when_empty. Retrieved 1/4 statements.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #34
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 10/20 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = [var_6, var_8]
    var_10 = {var_3: var_7, var_4: var_9}



# Parsed testcases at query #35
#--------------------------

# Partially parsed test_username_uppercase_tag_condition. Retrieved 3/9 statements.


def test_case_0():
    var_0 = []
    var_1 = 'testname'
    var_2 = 1950
    var_3 = 'U'
    var_4 = 'TESTNAME'



# Parsed testcases at query #36
#--------------------------

# Partially parsed test_surname_with_gender_separated_surnames. Retrieved 14/28 statements.


def test_case_0():
    var_0 = 'male'
    var_1 = 'female'
    var_2 = []
    var_3 = 'male'
    var_4 = 'female'
    var_5 = 'Smith'
    var_6 = 'Johnson'
    var_7 = [var_5, var_6]
    var_8 = 'Williams'
    var_9 = 'Brown'
    var_10 = [var_8, var_9]
    var_11 = {var_3: var_7, var_4: var_10}
    var_12 = 'surnames'
    var_13 = [var_12]
    var_14 = [var_5, var_6]



# Parsed testcases at query #37
#--------------------------

# Failed to parse test_nationality_returns_string.
# Failed to parse test_nationality_with_male_gender.
# Failed to parse test_nationality_with_female_gender.
# Partially parsed test_nationality_with_none_gender. Retrieved 1/5 statements.
# Failed to parse test_nationality_multiple_calls_return_strings.


def test_case_0():
    var_0 = []
    var_1 = None



# Parsed testcases at query #38
#--------------------------

# Failed to parse test_patronymic_returns_string_or_none.
# Failed to parse test_patronymic_with_male_gender.
# Failed to parse test_patronymic_with_female_gender.
# Partially parsed test_patronymic_with_none_gender. Retrieved 1/5 statements.
# Partially parsed test_patronymic_is_seeded. Retrieved 1/9 statements.


def test_case_0():
    var_0 = []
    var_1 = None

def test_case_0():
    var_0 = 42
    var_1 = []
    var_2 = []



